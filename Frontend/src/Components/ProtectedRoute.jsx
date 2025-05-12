import React from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import LoadingPage from "../Pages/LoadingPage";

const ProtectedRoute = ({ children }) => {
  const { isAuthenticated } = useAuth();

  if (isAuthenticated === null) {
    return <LoadingPage />;
  }

  return isAuthenticated ? children : <Navigate to="/auth" />;
};

export default ProtectedRoute;
