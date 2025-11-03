"""
Token utilities for authentication workflows.

Uses itsdangerous to generate signed, expiring tokens for operations such as
email verification. Tokens are tied to the `KIKA_SECRET_KEY` environment
variable so deployments can rotate credentials without code changes.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer


DEFAULT_EXPIRY_SECONDS = 48 * 60 * 60  # 48 hours by default
_EMAIL_SALT = "kika-email-verification"


class VerificationTokenError(Exception):
    """Base class for verification token problems."""


class VerificationTokenExpired(VerificationTokenError):
    """Raised when the token is valid but past its allowed lifetime."""

    def __init__(self, email: Optional[str]):
        self.email = email
        super().__init__("Verification token has expired.")


class VerificationTokenInvalid(VerificationTokenError):
    """Raised when a token cannot be decoded."""

    def __init__(self):
        super().__init__("Verification token is invalid.")


def _get_secret_key() -> str:
    key = os.getenv("KIKA_SECRET_KEY")
    if not key:
        raise RuntimeError(
            "KIKA_SECRET_KEY environment variable is required for verification tokens."
        )
    return key


@lru_cache(maxsize=1)
def _get_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(_get_secret_key())


def get_public_base_url() -> str:
    """
    Return the base URL used in verification links.
    """
    return os.getenv("KIKA_PUBLIC_BASE_URL", "http://localhost:8501").rstrip("/")


def generate_email_verification_token(email: str) -> str:
    """
    Create a signed token for the specified email address.
    """
    serializer = _get_serializer()
    return serializer.dumps(email, salt=_EMAIL_SALT)


def confirm_email_verification_token(token: str, max_age: Optional[int] = None) -> str:
    """
    Validate a verification token and return the embedded email address.

    Raises:
        VerificationTokenExpired: token expired (email returned via exception.email).
        VerificationTokenInvalid: token is malformed or has an invalid signature.
    """
    serializer = _get_serializer()
    max_age = max_age or DEFAULT_EXPIRY_SECONDS
    try:
        return serializer.loads(token, salt=_EMAIL_SALT, max_age=max_age)
    except SignatureExpired:
        try:
            email = serializer.loads(token, salt=_EMAIL_SALT)
        except BadSignature as exc:
            raise VerificationTokenInvalid() from exc
        raise VerificationTokenExpired(email)
    except BadSignature as exc:
        raise VerificationTokenInvalid() from exc
