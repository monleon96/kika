"""
Lightweight smoke checks for the email verification workflow.

Run with:
    python streamlit_app/testing/verify_flow.py
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
import sys
from datetime import datetime, timezone
import uuid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional for smoke script
    class _StreamlitStub:
        def __getattr__(self, name):
            raise RuntimeError(
                "Streamlit is required for UI features. Install streamlit to run the full app."
            )

    sys.modules["streamlit"] = _StreamlitStub()  # type: ignore

try:
    from utils.auth import (
        authenticate_user,
        mark_email_verified,
        register_user,
        resend_verification_email,
        _hash_password
    )
    from utils.db import init_db, db_cursor
    from utils.token_service import (
        VerificationTokenExpired,
        VerificationTokenInvalid,
        confirm_email_verification_token,
        generate_email_verification_token,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - developer guidance
    raise SystemExit(
        f"Missing dependency: {exc.name}. Install project requirements before running this script."
    ) from exc

from pathlib import Path
from dotenv import load_dotenv
# load project root .env (…/streamlit_app/testing/ -> up two levels)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


email = f"test.user+{uuid.uuid4().hex[:8]}@example.com"
password = "SuperSecret123!"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "kika-test.db"
        os.environ["KIKA_DB_PATH"] = str(db_path)
        os.environ["KIKA_SECRET_KEY"] = "test-secret-key"
        os.environ.pop("KIKA_SMTP_HOST", None)
        os.environ.pop("KIKA_SMTP_SENDER", None)

        init_db()

        email = "test.user@example.com"
        password = "SuperSecret123!"

        success, message = register_user("Test User", email, password)
        if not success:
            ml = message.lower()
            if "email service is not configured" in ml:
                # Create unverified user manually (since we disabled SMTP)
                with db_cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO users (email, full_name, password_hash, created_at, email_verified, is_active)
                        VALUES (?, ?, ?, ?, 0, 1)
                        """,
                        (
                            email,
                            "Test User",
                            _hash_password(password),
                            datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        ),
                    )
            elif "already exists" in ml:
                # Proceed — user is there; ensure it's unverified for the next assertions
                with db_cursor() as cur:
                    cur.execute("UPDATE users SET email_verified = 0 WHERE email = ?", (email,))
            else:
                assert success, f"Registration failed: {message}"


        success, user, message = authenticate_user(email, password)
        assert not success, "Login should be blocked before verification."
        assert "not verified" in message.lower()

        token = generate_email_verification_token(email)
        resolved_email = confirm_email_verification_token(token)
        assert resolved_email == email

        verified = mark_email_verified(email)
        assert verified, "Expected verification flag to be updated."

        success, user, message = authenticate_user(email, password)
        assert success, f"Login should succeed after verification: {message}"

        expired_token = generate_email_verification_token(email)
        time.sleep(1.5)
        try:
            confirm_email_verification_token(expired_token, max_age=1)
        except VerificationTokenExpired as exc:
            assert exc.email == email
        else:
            raise AssertionError("Expired token should raise VerificationTokenExpired.")

        tampered = token[:-1] + ("a" if token[-1] != "a" else "b")
        try:
            confirm_email_verification_token(tampered)
        except VerificationTokenInvalid:
            pass
        else:
            raise AssertionError("Tampered token should raise VerificationTokenInvalid.")

        resend_ok, resend_message = resend_verification_email(email)
        assert not resend_ok, "Resend should fail because email is already verified."
        assert "already verified" in resend_message.lower()

    print("Verification flow checks passed.")


if __name__ == "__main__":
    main()
