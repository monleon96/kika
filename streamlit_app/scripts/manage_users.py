#!/usr/bin/env python3
"""
Administrative helper for managing KIKA user accounts.

Usage:
    python streamlit_app/scripts/manage_users.py list-users
    python streamlit_app/scripts/manage_users.py create-user --name "Jane" --email jane@example.com
    python streamlit_app/scripts/manage_users.py set-password --email jane@example.com
    python streamlit_app/scripts/manage_users.py deactivate-user --email jane@example.com
    python streamlit_app/scripts/manage_users.py send-verification --email jane@example.com
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.auth import (  # noqa: E402  (import after sys.path manipulation)
    list_all_users,
    register_user,
    resend_verification_email,
    set_user_active,
    update_user_password,
)


def _prompt_password(confirm: bool = True) -> str:
    while True:
        first = getpass.getpass("Password: ")
        if confirm:
            second = getpass.getpass("Confirm password: ")
            if first != second:
                print("Passwords do not match. Try again.")
                continue
        return first


def cmd_list_users(_: argparse.Namespace) -> int:
    users = list_all_users()
    if not users:
        print("No users found.")
        return 0

    print(f"{'Email':40} {'Name':25} {'Active':7} {'Verified':8} {'Created'} {'Last login'}")
    print("-" * 120)
    for user in users:
        email = user["email"] or "(guest)"
        name = user["full_name"]
        active = "yes" if user["is_active"] else "no"
        verified = "yes" if user.get("email_verified") else "no"
        created = user["created_at"] or "-"
        last_login = user["last_login"] or "-"
        print(f"{email:40} {name:25} {active:7} {verified:8} {created:19} {last_login}")
    return 0


def cmd_create_user(args: argparse.Namespace) -> int:
    password = args.password or _prompt_password()
    success, message = register_user(args.name, args.email, password)
    print(message)
    return 0 if success else 1


def cmd_send_verification(args: argparse.Namespace) -> int:
    success, message = resend_verification_email(args.email)
    print(message)
    return 0 if success else 1


def cmd_set_password(args: argparse.Namespace) -> int:
    password = args.password or _prompt_password()
    success, message = update_user_password(args.email, password)
    print(message)
    return 0 if success else 1


def cmd_activate(args: argparse.Namespace, activate: bool) -> int:
    success, message = set_user_active(args.email, activate)
    print(message)
    return 0 if success else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage KIKA user accounts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sp_list = subparsers.add_parser("list-users", help="Show all registered users.")
    sp_list.set_defaults(func=cmd_list_users)

    sp_create = subparsers.add_parser("create-user", help="Create a new user account.")
    sp_create.add_argument("--name", required=True, help="Full name of the user.")
    sp_create.add_argument("--email", required=True, help="Email address.")
    sp_create.add_argument(
        "--password",
        help="Password (omit to be prompted securely).",
    )
    sp_create.set_defaults(func=cmd_create_user)

    sp_password = subparsers.add_parser("set-password", help="Reset a user's password.")
    sp_password.add_argument("--email", required=True, help="Email address.")
    sp_password.add_argument(
        "--password",
        help="New password (omit to be prompted securely).",
    )
    sp_password.set_defaults(func=cmd_set_password)

    sp_deactivate = subparsers.add_parser("deactivate-user", help="Deactivate a user account.")
    sp_deactivate.add_argument("--email", required=True, help="Email address.")
    sp_deactivate.set_defaults(
        func=lambda ns: cmd_activate(ns, activate=False)
    )

    sp_activate = subparsers.add_parser("activate-user", help="Reactivate a user account.")
    sp_activate.add_argument("--email", required=True, help="Email address.")
    sp_activate.set_defaults(
        func=lambda ns: cmd_activate(ns, activate=True)
    )

    sp_send = subparsers.add_parser("send-verification", help="Send or resend a verification email.")
    sp_send.add_argument("--email", required=True, help="Email address.")
    sp_send.set_defaults(func=cmd_send_verification)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
