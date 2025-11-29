#!/usr/bin/env python3
"""Test backend connection"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Now import api_client
from utils.api_client import get_backend_url, check_backend_health

print("=" * 50)
print("🔍 Testing Backend Connection")
print("=" * 50)
print()

backend_url = get_backend_url()
print(f"Backend URL: {backend_url}")
print()

print("Testing health check...")
is_healthy = check_backend_health()

if is_healthy:
    print("✅ Backend is healthy and reachable!")
else:
    print("❌ Backend is not reachable")
    sys.exit(1)

print()
print("=" * 50)
print("✅ All tests passed!")
print("=" * 50)
