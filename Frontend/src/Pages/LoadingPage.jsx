import React from "react";
import { CircularProgress, Box, Typography } from "@mui/material";

const LoadingPage = () => {
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh",
        background: "linear-gradient(130deg, #000 40%, #FE5F78 100%)",
        justifyContent: "center",
        alignItems: "center",
        textAlign: "center",
      }}
    >
      <CircularProgress size={60} sx={{ color: "white" }} />
      <Typography
        variant="h4"
        sx={{
          color: "white",
          marginTop: 3,
          fontWeight: "bold",
          textShadow: "1px 1px 4px rgba(0,0,0,0.6)",
        }}
      >
        Loading, Please Wait...
      </Typography>
    </Box>
  );
};

export default LoadingPage;
