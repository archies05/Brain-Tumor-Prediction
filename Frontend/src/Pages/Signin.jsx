import React, { useState } from "react";
import {
  Box,
  TextField,
  Button,
  Typography,
  Link,
  CircularProgress,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import { account } from "../appwrite";

const Signin = ({ switchToSignUp }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const navigate = useNavigate();

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const session = await account.createEmailPasswordSession(email, password);
      // console.log("User signed in:", session);

      navigate("/");
    } catch (err) {
      console.error("Error during sign-in:", err);
      setError(err.message || "Login failed. Please check your credentials.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      component="form"
      sx={{
        backgroundColor: "#fff",
        borderRadius: "25px ",
        px: 4,
        pt: 4,
        pb: 6,
      }}
      onSubmit={handleSubmit}
    >
      <TextField
        fullWidth
        label="Email"
        variant="standard"
        margin="normal"
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        sx={{
          "& .MuiInput-underline:before": {
            borderBottomColor: "rgba(0, 0, 0, 0.42)",
          },
          "& .MuiInput-underline:after": {
            borderBottomColor: "#FE5F78",
          },
        }}
        InputLabelProps={{
          style: { color: "#888" },
        }}
        InputProps={{
          style: { color: "black" },
          disableUnderline: false,
        }}
      />
      <TextField
        fullWidth
        label="Password"
        variant="standard"
        margin="normal"
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        sx={{
          "& .MuiInput-underline:before": {
            borderBottomColor: "rgba(0, 0, 0, 0.42)",
          },
          "& .MuiInput-underline:after": {
            borderBottomColor: "#FE5F78",
          },
        }}
        InputLabelProps={{
          style: { color: "#888" },
        }}
        InputProps={{
          style: { color: "black" },
          disableUnderline: false,
        }}
      />

      {error && (
        <Typography color="error" variant="body2" textAlign="center" mt={2}>
          {error}
        </Typography>
      )}

      <Button
        type="submit"
        fullWidth
        variant="contained"
        sx={{
          marginTop: 2,
          backgroundColor: " #FE5F78",
          boxShadow: 0,
          ":hover": { backgroundColor: "rgb(180, 70, 86)", boxShadow: 0 },
        }}
        disabled={loading}
      >
        {loading ? <CircularProgress size={24} color="inherit" /> : "Sign In"}
      </Button>

      <Typography
        variant="body2"
        textAlign="center"
        marginTop={3}
        color="text.secondary"
      >
        Don’t have an account?{" "}
        <Link
          href="#"
          underline="none"
          sx={{ color: " #FE5F78", fontWeight: 600 }}
          onClick={switchToSignUp}
        >
          Sign Up
        </Link>
      </Typography>
    </Box>
  );
};

export default Signin;
