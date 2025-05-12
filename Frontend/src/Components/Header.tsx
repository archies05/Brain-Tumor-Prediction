import React, { useState, useEffect } from "react";
import { Link as RouterLink, useNavigate } from "react-router-dom";
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Container,
  Box,
  Avatar,
  Popover,
  MenuItem,
  ListItemText,
} from "@mui/material";
import { Logout } from "@mui/icons-material";
import { useAuth } from "../context/AuthContext"; // Assuming you're using useAuth to get user details
import { account } from "../appwrite";

const Header: React.FC = () => {
  const navigate = useNavigate();
  const [scrolled, setScrolled] = useState(false);
  const [anchorEl, setAnchorEl] = useState(null); // State to control the Popover

  // Get user information from context
  const { user, isAuthenticated, setUser } = useAuth();

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 50) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };

    window.addEventListener("scroll", handleScroll);

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  const handleLogout = async () => {
    try {
      await account.deleteSession("current");

      setUser(null);
      navigate("/auth");
    } catch (error) {
      console.error("Error during logout:", error);
    }
  };

  const handleProfileClick = (event) => {
    setAnchorEl(event.currentTarget); // Open Popover on profile click
  };

  const handleCloseProfile = () => {
    setAnchorEl(null); // Close Popover
  };

  const open = Boolean(anchorEl); // Determines if the Popover is open
  const id = open ? "profile-popover" : undefined; // Popover id

  return (
    <AppBar
      position="fixed"
      sx={{
        background: scrolled ? "rgba(0, 0, 0, 0.7)" : "transparent",
        backdropFilter: scrolled ? "blur(5px)" : "none",
        boxShadow: scrolled ? "0 4px 6px rgba(0, 0, 0, 0.1)" : "none",
        width: "100%",
        zIndex: 1000,
      }}
    >
      <Container maxWidth="lg">
        <Toolbar disableGutters>
          <Typography
            variant="h4"
            component={RouterLink}
            to="/"
            sx={{
              flexGrow: 1,
              textDecoration: "none",
              color: "white",
              fontWeight: "bold",
              textShadow: "1px 1px 4px rgba(0,0,0,0.6)",
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

          <Box sx={{ display: "flex" }}>
            <Button
              color="inherit"
              component={RouterLink}
              to="/"
              sx={{
                color: "white",
                fontWeight: "bold",
                textTransform: "capitalize",
                fontSize: "16px",
                mx: 1,
                "&:hover": {
                  backgroundColor: "rgba(255, 255, 255, 0.2)",
                },
              }}
            >
              Home
            </Button>
            <Button
              color="inherit"
              component={RouterLink}
              to="/upload"
              sx={{
                color: "white",
                fontWeight: "bold",
                textTransform: "capitalize",
                fontSize: "16px",
                mx: 1,
                "&:hover": {
                  backgroundColor: "rgba(255, 255, 255, 0.2)",
                },
              }}
            >
              Upload Scan
            </Button>
            <Button
              color="inherit"
              component={RouterLink}
              to="/segmentation-image"
              sx={{
                color: "white",
                fontWeight: "bold",
                textTransform: "capitalize",
                fontSize: "16px",
                mx: 1,
                "&:hover": {
                  backgroundColor: "rgba(255, 255, 255, 0.2)",
                },
              }}
            >
              Check Segmentation
            </Button>

            {isAuthenticated && user ? (
              <Box display="flex" alignItems="center">
                <Avatar
                  sx={{
                    width: 30,
                    height: 30,
                    backgroundColor: "#FE5F78",
                    marginRight: 1,
                    cursor: "pointer", // Makes it clickable
                  }}
                  onClick={handleProfileClick} // Opens the profile popover
                >
                  {user.name?.charAt(0).toUpperCase()}
                </Avatar>

                <Typography
                  variant="h6"
                  sx={{
                    color: "white",
                    fontWeight: "bold",
                    marginRight: 2,
                    textTransform: "capitalize",
                    cursor: "pointer", // Makes it clickable
                  }}
                  onClick={handleProfileClick} // Opens the profile popover
                >
                  {user.name || "User"}
                </Typography>

                {/* Profile Popover */}
                <Popover
                  id={id}
                  open={open}
                  anchorEl={anchorEl}
                  onClose={handleCloseProfile}
                  anchorOrigin={{
                    vertical: "bottom",
                    horizontal: "center",
                  }}
                  transformOrigin={{
                    vertical: "top",
                    horizontal: "center",
                  }}
                  sx={{
                    "& .MuiPopover-paper": {
                      backdropFilter: "blur(10px)",
                      backgroundColor: "rgba(255, 255, 255, 0.1)", // Slight transparent background for popover
                      color: "white",
                      boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)", // Optional shadow for the popover
                      mt: 1,
                      px: 2,
                    },
                  }}
                >
                  <Box>
                    <MenuItem onClick={handleLogout}>
                      <ListItemText primary="Logout" />
                    </MenuItem>
                  </Box>
                </Popover>
              </Box>
            ) : (
              <Box>
                <Button
                  color="inherit"
                  component={RouterLink}
                  to="/auth"
                  sx={{
                    color: "white",
                    fontWeight: "bold",
                    textTransform: "capitalize",
                    fontSize: "16px",
                    mx: 1,
                    "&:hover": {
                      backgroundColor: "rgba(255, 255, 255, 0.2)",
                    },
                  }}
                >
                  Login
                </Button>
              </Box>
            )}
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
};

export default Header;
