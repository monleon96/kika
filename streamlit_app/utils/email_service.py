"""
Outbound email utilities for the Streamlit UI.

Minimal SMTP integration so the application can send transactional emails
(e.g., welcome messages, password resets). Configuration comes from env vars.

Env (safe defaults in comments):
  # Required to enable email
  KIKA_SMTP_HOST=...
  KIKA_SMTP_SENDER="KIKA <tuusuario@gmail.com>"

  # Common/optional
  KIKA_SMTP_PORT=587
  KIKA_SMTP_USERNAME=...
  KIKA_SMTP_PASSWORD=...
  KIKA_SMTP_USE_TLS=true      # STARTTLS (port 587) - default
  KIKA_SMTP_USE_SSL=false     # SMTPS (port 465). If true, overrides USE_TLS.
  KIKA_SMTP_REPLY_TO="KIKA Support <tucorreo@gmail.com>"
  # Envelope sender for bounces (MAIL FROM). If unset, we reuse KIKA_SMTP_SENDER.
  KIKA_SMTP_ENVELOPE_SENDER="no-reply@tudominio.com"
"""

from __future__ import annotations

import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Optional


class EmailServiceError(RuntimeError):
    """Raised when outbound email cannot be sent due to configuration or SMTP failures."""


def _env_bool(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class EmailConfig:
    host: str
    port: int
    username: Optional[str]
    password: Optional[str]
    use_starttls: bool
    use_ssl: bool
    sender: str
    reply_to: Optional[str]
    envelope_sender: Optional[str]


def load_email_config() -> Optional[EmailConfig]:
    """
    Read SMTP settings from environment variables. Returns None when the app
    is not configured for outbound email (missing host or sender).
    """
    host = os.getenv("KIKA_SMTP_HOST")
    sender = os.getenv("KIKA_SMTP_SENDER")
    if not host or not sender:
        return None

    # SSL takes precedence over STARTTLS if explicitly enabled
    use_ssl = _env_bool("KIKA_SMTP_USE_SSL", False)
    use_starttls = _env_bool("KIKA_SMTP_USE_TLS", not use_ssl)

    # Default port: 465 for SSL, 587 otherwise
    default_port = 465 if use_ssl else 587
    port = int(os.getenv("KIKA_SMTP_PORT", str(default_port)))

    username = os.getenv("KIKA_SMTP_USERNAME") or None
    password = os.getenv("KIKA_SMTP_PASSWORD") or None
    reply_to = os.getenv("KIKA_SMTP_REPLY_TO") or None
    envelope_sender = os.getenv("KIKA_SMTP_ENVELOPE_SENDER") or None

    return EmailConfig(
        host=host,
        port=port,
        username=username,
        password=password,
        use_starttls=use_starttls,
        use_ssl=use_ssl,
        sender=sender,
        reply_to=reply_to,
        envelope_sender=envelope_sender,
    )


def send_email(
    to_address: str,
    subject: str,
    body: str,
    *,
    html: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
) -> bool:
    """
    Send an email using the configured SMTP server.

    Returns:
      True  -> email attempted/sent
      False -> email disabled (no config)

    Raises:
      EmailServiceError on configuration or SMTP failures.
    """
    config = load_email_config()
    if config is None:
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["To"] = to_address
    msg["From"] = config.sender
    if config.reply_to:
        msg["Reply-To"] = config.reply_to
    if headers:
        for k, v in headers.items():
            # Avoid overriding standard headers already set
            if k.lower() not in {"subject", "to", "from", "reply-to"}:
                msg[k] = v

    msg.set_content(body)
    if html:
        msg.add_alternative(html, subtype="html")

    # Envelope sender controls MAIL FROM (affects Return-Path/bounces)
    envelope_from = config.envelope_sender or config.sender

    try:
        if config.use_ssl:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(config.host, config.port, context=context, timeout=20) as smtp:
                if config.username and config.password:
                    smtp.login(config.username, config.password)
                smtp.send_message(msg, from_addr=envelope_from)
        else:
            with smtplib.SMTP(config.host, config.port, timeout=20) as smtp:
                if config.use_starttls:
                    context = ssl.create_default_context()
                    smtp.starttls(context=context)
                if config.username and config.password:
                    smtp.login(config.username, config.password)
                smtp.send_message(msg, from_addr=envelope_from)
    except Exception as exc:  # noqa: BLE001
        # Surface a cleaner message but keep original for debugging
        raise EmailServiceError(f"SMTP error: {exc}") from exc

    return True


def send_welcome_email(full_name: str, to_address: str) -> bool:
    """
    Dispatch a simple welcome email to new users.

    Returns True if an email was sent, False when email is not configured.
    """
    subject = "Welcome to KIKA"
    body = (
        f"Hi {full_name},\n\n"
        "Thanks for creating a KIKA account. Your nuclear-data preferences are now synced "
        "whenever you sign in.\n\n"
        "Happy analyzing!\n"
        "— The MCNPy Team"
    )
    html_body = (
        f"<p>Hi {full_name},</p>"
        "<p>Thanks for creating a <strong>KIKA</strong> account. Your nuclear-data preferences "
        "are now synced whenever you sign in.</p>"
        "<p>Happy analyzing!<br/>— The MCNPy Team</p>"
    )
    # Example of adding a List-Unsubscribe header (some providers like it)
    headers = {
        # Replace with a real manage-preferences URL when you have one
        "List-Unsubscribe": "<mailto:no-reply@invalid.example>",
    }
    return send_email(to_address, subject, body, html=html_body, headers=headers)
