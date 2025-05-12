import React, { useState } from "react";
import { Box, Typography, Tabs, Tab } from "@mui/material";
import SignUp from "./SignUp";
import Signin from "./Signin";

const Auth = () => {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: { xs: "column", md: "row" },
        height: "100vh",
        fontFamily: "'Poppins', sans-serif",
      }}
    >
      <Box
        sx={{
          flex: 1,
        }}
      >
        <Typography
          variant="h4"
          sx={{
            flexGrow: 1,
            textDecoration: "none",
            color: "white",
            fontWeight: "bold",
            textShadow: "1px 1px 4px rgba(0,0,0,0.6)",
            ml: 3,
            mt: 2,
          }}
        >
          Brain Vision
          <span
            style={{
              color: "rgb(141, 45, 59)",
              borderRadius: "50%",
              fontSize: "40px",
            }}
          >
            .
          </span>
        </Typography>
        <Box
          sx={{
            flex: 1,
            // background:
            //   "linear-gradient(to bottom right, rgb(106, 199, 241),rgba(25, 60, 173, 0.72))",
            display: { xs: "none", md: "flex" },
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            color: "#fff",
          }}
        >
          <Box
            component="img"
            src="https://i.postimg.cc/C5QKGjSz/Cancer-brain-tumor-treatment-GBM-removebg-preview.png"
            sx={{
              width: "100%",
              mt: 3,
              borderRadius: "16px",
              animation: "upDown 2s infinite ease-in-out",
              "@keyframes upDown": {
                "0%": { transform: "translateY(0)" },
                "50%": { transform: "translateY(-20px)" },
                "100%": { transform: "translateY(0)" },
              },
            }}
          />
        </Box>
      </Box>

      <Box
        sx={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Box
          sx={{
            width: "100%",
            maxWidth: "400px",
            backgroundColor: "#eee",
            borderRadius: "25px",
            boxShadow: 3,
          }}
        >
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            variant="fullWidth"
            sx={{
              marginBottom: 3,
              "& .MuiTabs-indicator": {
                backgroundColor: "rgb(128, 42, 55)",
              },
              "& .MuiTab-root": {
                textTransform: "none",
                fontWeight: "bold",
                color: "gray",
                borderTopRightRadius: "25px",
                borderTopLeftRadius: "25px",
              },
              "& .MuiTab-root.Mui-selected": {
                color: "rgb(128, 42, 55)",
                borderTopRightRadius: "25px",
                borderTopLeftRadius: "25px",
              },
            }}
          >
            <Tab label="Sign Up" />
            <Tab label="Sign In" />
          </Tabs>

          {tabValue === 0 ? (
            <SignUp switchToSignIn={() => setTabValue(1)} />
          ) : (
            <Signin switchToSignUp={() => setTabValue(0)} />
          )}
        </Box>
      </Box>
    </Box>
  );
};

export default Auth;
