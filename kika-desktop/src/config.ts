/**
 * Application Configuration
 */

// Backend API URL
// Use environment variable or default to production backend
export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'https://kika-backend.onrender.com';

// MCNPy Processing Server URL (local Python server)
export const MCNPY_SERVER_URL = import.meta.env.VITE_MCNPY_SERVER_URL || 'http://localhost:8001';

// API endpoints
export const API_ENDPOINTS = {
  health: '/healthz',
  login: '/login',
  register: '/register',
  users: '/users',
  verify: '/verify',
  passwordForgot: '/password/forgot',
  passwordReset: '/password/reset',
} as const;

export default {
  BACKEND_URL,
  API_ENDPOINTS,
};
