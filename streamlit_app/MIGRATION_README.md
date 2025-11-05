# KIKA Streamlit App - Backend Migration

## Overview

The KIKA Streamlit application has been refactored to use the **kika-backend** FastAPI server for all authentication and user management operations, replacing the previous embedded SQLite-based authentication system.

## What Changed

### ✅ New Files Added

1. **`utils/api_client.py`** - HTTP client for communicating with kika-backend API
2. **`utils/backend_auth.py`** - Authentication module using backend API (replaces old `auth.py`)
3. **`.env.example`** - Environment configuration template

### 🗑️ Deprecated Files Removed

The following files have been **removed** as they are now redundant:

1. **`utils/auth.py`** - Replaced by `utils/backend_auth.py`
2. **`utils/db.py`** - SQLite database access (no longer needed)
3. **`utils/email_service.py`** - SMTP email sending (handled by backend)
4. **`utils/token_service.py`** - Token generation/validation (handled by backend)
5. **`scripts/manage_users.py`** - User management CLI (use backend admin API)
6. **`testing/verify_flow.py`** - Tests for old auth system

### 📝 Modified Files

1. **`KIKA.py`** - Updated imports to use `backend_auth`
2. **`pages/1_📊_ACE_Viewer.py`** - Updated imports
3. **`pages/2_📈_ENDF_Viewer.py`** - Updated imports
4. **`pages/3_🔧_NJOY_Processing.py`** - Updated imports
5. **`pages/5_⚙️_Settings.py`** - Updated imports
6. **`utils/__init__.py`** - Updated exports to use `backend_auth`
7. **`utils/user_settings.py`** - Removed SQLite dependency
8. **`requirements.txt`** - Added `requests`, removed auth-specific packages

## Configuration

### Environment Variables

Create a `.env` file in the `streamlit_app` directory:

```bash
# Backend API URL (required)
KIKA_BACKEND_URL=http://localhost:8000

# Admin API Key (optional, for admin operations)
KIKA_ADMIN_KEY=your-admin-key-here
```

### Backend Setup

Before running the Streamlit app, ensure the kika-backend is running:

```bash
# In the kika-backend directory
cd /path/to/kika-backend

# Install dependencies
pip install -r requirements.txt

# Configure backend .env
cp .env.sample .env
# Edit .env with your settings

# Run migrations
alembic upgrade head

# Start the backend server
uvicorn app:app --reload
```

The backend will be available at `http://localhost:8000` by default.

## API Endpoints Used

The Streamlit app now communicates with these kika-backend endpoints:

### Authentication
- `POST /register` - Create new user account
- `POST /login` - Authenticate user
- `GET /verify?token=...` - Verify email address
- `POST /password/forgot` - Request password reset
- `POST /password/reset` - Reset password with token

### User Management
- `GET /users/{email}` - Get user status
- `POST /admin/users/create` - Admin: Create user
- `POST /admin/users/deactivate` - Admin: Deactivate user
- `GET /admin/users/list` - Admin: List all users

### Metrics (optional)
- `POST /metrics/event` - Track analytics events

### Health
- `GET /health` - Check backend availability

## How It Works

### Old System (Removed)
```
Streamlit App → SQLite DB
            ↓
    Email via SMTP
```

### New System (Current)
```
Streamlit App → API Client → kika-backend → PostgreSQL
                                        ↓
                                Email via SMTP
```

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export KIKA_BACKEND_URL=http://localhost:8000

# Or use .env file (recommended)
cp .env.example .env
# Edit .env with your settings

# Run the app
streamlit run KIKA.py
```

## User Settings Storage

**Current Status:** User settings (plot preferences, NJOY config, etc.) are stored in **session state only** and reset when the session ends.

**Future Enhancement:** A backend endpoint should be added to persist user settings to the database.

## Admin Operations

### Using the API

Admin operations require the `X-Admin-Key` header:

```python
from utils.api_client import admin_list_users, admin_create_user, admin_deactivate_user

# List users
users = admin_list_users(limit=100)

# Create user
success, msg = admin_create_user("user@example.com", "password123", verified=True)

# Deactivate user
success, msg = admin_deactivate_user("user@example.com")
```

### Using the Backend CLI

The kika-backend provides a CLI for user management:

```bash
cd /path/to/kika-backend

# List users
python cli.py list-users

# Create user
python cli.py create-user --email user@example.com --password pass123

# Deactivate user
python cli.py deactivate-user --email user@example.com
```

See `kika-backend/CLI_USAGE.md` for full documentation.

## Migration Checklist

- [x] Create API client module
- [x] Create new backend auth module
- [x] Update all imports across pages
- [x] Remove deprecated authentication files
- [x] Remove SQLite database dependencies
- [x] Update user_settings to work without SQLite
- [x] Update requirements.txt
- [x] Create environment configuration template
- [x] Document changes in README

## Troubleshooting

### "Unable to connect to backend"

1. Check that kika-backend is running: `curl http://localhost:8000/health`
2. Verify `KIKA_BACKEND_URL` environment variable
3. Check backend logs for errors

### "Invalid credentials"

1. Ensure user is registered and email is verified
2. Check backend database: `python cli.py list-users`
3. Verify user is active and verified

### "Too many requests"

The backend has rate limiting. Wait a moment and try again.

### Settings not persisting

This is expected - user settings are currently session-only. Backend persistence needs to be implemented.

## Next Steps

### For Deployment

1. **Deploy kika-backend** to a server or cloud platform
2. **Update `KIKA_BACKEND_URL`** to point to deployed backend
3. **Configure SMTP** in backend for email sending
4. **Set up PostgreSQL** database for production
5. **Add SSL/TLS** for secure communication

### Future Enhancements

1. **User Settings API** - Add backend endpoints for persistent user settings
2. **Session Management** - Implement JWT tokens for better session handling
3. **OAuth Integration** - Add Google/GitHub login options
4. **Password Strength** - Add password requirements UI
5. **Profile Management** - Allow users to update their profile info

## Support

- Backend documentation: See `kika-backend/README.md`
- Backend API docs: `http://localhost:8000/docs` (when backend is running)
- Issues: Report on GitHub
