import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { BACKEND_URL } from '../config';

export interface User {
  email: string;
  verified: boolean;
  is_active: boolean;
  is_guest?: boolean;
}

interface AuthContextType {
  user: User | null;
  login: (email: string, password: string) => Promise<{ success: boolean; message: string }>;
  loginAsGuest: () => void;
  logout: () => void;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load user from localStorage on mount
  useEffect(() => {
    const storedUser = localStorage.getItem('kika_user');
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser));
      } catch (e) {
        localStorage.removeItem('kika_user');
      }
    }
    setIsLoading(false);
  }, []);

  const login = async (email: string, password: string) => {
    try {
      const response = await fetch(`${BACKEND_URL}/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      if (response.ok) {
        // Fetch user details
        const userResponse = await fetch(`${BACKEND_URL}/users/${email}`);
        if (userResponse.ok) {
          const userData: User = await userResponse.json();
          setUser(userData);
          localStorage.setItem('kika_user', JSON.stringify(userData));
          return { success: true, message: 'Signed in successfully' };
        }
      }

      return { success: false, message: 'Invalid credentials' };
    } catch (error) {
      return { success: false, message: 'Connection error. Is the backend running?' };
    }
  };

  const loginAsGuest = () => {
    const guestUser: User = {
      email: 'guest@local',
      verified: true,
      is_active: true,
      is_guest: true,
    };
    setUser(guestUser);
    localStorage.setItem('kika_user', JSON.stringify(guestUser));
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('kika_user');
    // Clean up all session data including saved plot configurations
    // This is especially important for guest users
    localStorage.removeItem('kikaAcePlotterConfigs'); // ACE plot configs
    localStorage.removeItem('kikaEndfViewerConfigs'); // ENDF plot configs
    localStorage.removeItem('workspaceFiles'); // Legacy file storage
  };

  return (
    <AuthContext.Provider value={{ user, login, loginAsGuest, logout, isLoading }}>
      {children}
    </AuthContext.Provider>
  );
};
