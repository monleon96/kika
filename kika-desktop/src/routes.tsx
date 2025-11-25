import { Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Home } from './pages/Home';
import { Login } from './pages/Login';
import { ACEViewer } from './pages/ACEViewer';
import { ENDFViewer } from './pages/ENDFViewer';
import { useAuth } from './contexts/AuthContext';

// Protected route wrapper
interface ProtectedRouteProps {
  children: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { user } = useAuth();
  
  if (!user) {
    return <Navigate to="/login" replace />;
  }
  
  return <>{children}</>;
};

export const AppRoutes: React.FC = () => {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route index element={<Home />} />
        <Route path="ace-viewer" element={<ACEViewer />} />
        <Route path="endf-viewer" element={<ENDFViewer />} />
        {/* More routes will be added here as we migrate pages */}
      </Route>
    </Routes>
  );
};
