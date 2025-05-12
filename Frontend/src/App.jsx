import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useLocation,
} from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import HomePage from "./Pages/HomePage";
import NotFound from "./Pages/NotFound";
import Auth from "./Pages/Auth";
import UploadPage from "./Pages/UploadPage";
import { createTheme, CssBaseline, ThemeProvider } from "@mui/material";
import Header from "./Components/Header";
import Footer from "./Components/Footer";
import ProtectedRoute from "./Components/ProtectedRoute";
import SegmentationUpload from "./Pages/SegmanationUpload";

const theme = createTheme({
  palette: {
    primary: {
      main: "#1976d2",
    },
    secondary: {
      main: "#00bcd4",
    },
  },
});

function Layout({ children }) {
  const location = useLocation();
  const hideHeaderFooter = location.pathname === "/auth";

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh",
        background: "linear-gradient(130deg, #000 40%, #FE5F78 100%)",
      }}
    >
      {!hideHeaderFooter && <Header />}
      <main style={{ flexGrow: 1 }}>{children}</main>
      {!hideHeaderFooter && <Footer />}
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      {" "}
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <Layout>
            <Routes>
              <Route
                path="/"
                element={
                  <ProtectedRoute>
                    <HomePage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/upload"
                element={
                  <ProtectedRoute>
                    <UploadPage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/segmentation-image"
                element={
                  <ProtectedRoute>
                    <SegmentationUpload />
                  </ProtectedRoute>
                }
              />
              <Route path="/auth" element={<Auth />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Layout>
        </Router>
      </ThemeProvider>
    </AuthProvider>
  );
}

export default App;
