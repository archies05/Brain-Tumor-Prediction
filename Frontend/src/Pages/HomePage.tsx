import React, { useEffect, useState } from "react";
import { Link as RouterLink } from "react-router-dom";
import { motion } from "framer-motion";

import {
  Container,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  Box,
  useTheme,
  useMediaQuery,
  Fade,
  Grow,
  CardMedia,
} from "@mui/material";
import {
  ArrowForward,
  CloudUpload,
  Psychology,
  Assessment,
  Speed,
  Accessibility,
  VerifiedUser,
} from "@mui/icons-material";

const HomePage: React.FC = () => {
  const theme = useTheme();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down("sm"));
  const fullText =
    "Harness the power of artificial intelligence for advanced brain tumor detection. Quick, accurate, and non-invasive analysis at your fingertips.";

  const [currentText, setCurrentText] = useState("");
  const [charIndex, setCharIndex] = useState(0);

  useEffect(() => {
    const typingSpeed = 70; // Speed at which each character is typed (in ms)
    const resetSpeed = 2000; // Delay before resetting the text (in ms)

    const typeWriter = setInterval(() => {
      if (charIndex < fullText.length) {
        setCurrentText((prev) => prev + fullText[charIndex]);
        setCharIndex((prev) => prev + 1);
      } else {
        clearInterval(typeWriter);
        setTimeout(() => {
          setCurrentText(""); // Clear the text after the full sentence is typed
          setCharIndex(0); // Reset the character index to start over
        }, resetSpeed); // Delay before starting over
      }
    }, typingSpeed);

    return () => clearInterval(typeWriter); // Clean up interval when the component unmounts
  }, [charIndex]);

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <Container
        maxWidth="lg"
        sx={{
          flexGrow: 1,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          py: 8,
        }}
      >
        <Fade in={true} timeout={1000}>
          <Box textAlign="center" mb={10}>
            <motion.div
              initial={{ opacity: 0, y: -50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1 }}
            >
              <Typography
                variant="h2"
                component="h1"
                gutterBottom
                sx={{
                  color: "white",
                  fontWeight: "bold",
                  textShadow: "2px 2px 8px rgba(0,0,0,0.6)",
                }}
              >
                Welcome to Brain Vision
              </Typography>
            </motion.div>

            <Typography
              variant="h5"
              paragraph
              sx={{
                color: "rgba(255, 255, 255, 0.9)",
                maxWidth: "700px",
                margin: "0 auto",
                textShadow: "1px 1px 4px rgba(0,0,0,0.4)",
              }}
            >
              {currentText}
            </Typography>
            <Button
              component={RouterLink}
              to="/upload"
              variant="contained"
              size="large"
              endIcon={<ArrowForward />}
              sx={{
                mt: 4,
                px: 5,
                py: 2,
                fontSize: "1.2rem",
                bgcolor: "#ffffff",
                color: " #4A1C23",
                borderRadius: "30px",
                "&:hover": {
                  bgcolor: "#f5f5f5",
                },
              }}
            >
              Start Scanning
            </Button>
          </Box>
        </Fade>

        <Grid container spacing={6} mb={10}>
          <Grid item xs={12} md={6}>
            <Typography
              variant="h4"
              gutterBottom
              sx={{
                color: "white",
                textShadow: "1px 1px 4px rgba(0,0,0,0.4)",
              }}
            >
              How It Works
            </Typography>
            <Grid container spacing={3}>
              {[
                { icon: <CloudUpload />, text: "Upload your brain MRI scan" },
                { icon: <Psychology />, text: "Our AI analyzes the image" },
                {
                  icon: <Speed />,
                  text: "Segmentation and classification of tumor types",
                },
                {
                  icon: <Assessment />,
                  text: "Receive detailed results and insights",
                },
              ].map((step, index) => (
                <Grow in={true} timeout={(index + 1) * 500} key={index}>
                  <Grid item xs={12}>
                    <Card
                      elevation={4}
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        borderRadius: "15px",
                        transition: "transform 0.3s ease-in-out",
                        "&:hover": {
                          transform: "scale(1.03)",
                        },
                      }}
                    >
                      <CardContent
                        sx={{
                          display: "flex",
                          alignItems: "center",
                        }}
                      >
                        <Box sx={{ mr: 2, color: "#4A1C23", fontSize: 40 }}>
                          {step.icon}
                        </Box>
                        <Typography
                          variant="body1"
                          sx={{ fontSize: "1.1rem", fontWeight: "bold" }}
                        >
                          {step.text}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grow>
              ))}
            </Grid>
          </Grid>

          <Grid
            item
            xs={12}
            md={6}
            sx={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
            }}
          >
            <Fade in={true} timeout={1500}>
              <motion.div
                initial={{ opacity: 0, y: 100 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 1 }}
              >
                <CardMedia
                  component="img"
                  image="https://i.postimg.cc/C5QKGjSz/Cancer-brain-tumor-treatment-GBM-removebg-preview.png"
                  alt="Brain scan illustration"
                  sx={{
                    width: "100%",
                    maxWidth: 9000,
                    height: "auto",
                    animation: "upDown 2s infinite ease-in-out",
                    "@keyframes upDown": {
                      "0%": { transform: "translateY(0)" },
                      "50%": { transform: "translateY(-20px)" },
                      "100%": { transform: "translateY(0)" },
                    },
                  }}
                />
              </motion.div>
            </Fade>
          </Grid>
        </Grid>

        <Card
          sx={{
            mb: 10,
            py: 5,
            px: 3,
            borderRadius: "20px",
            background: "transparent",
            boxShadow: "inset 0px 2px 40px rgba(0, 0, 0, 0.3)",
          }}
        >
          <CardContent>
            <Typography
              variant="h4"
              align="center"
              gutterBottom
              color="#fff"
              sx={{
                textShadow: "1px 1px 4px rgba(0,0,0,0.3)",
              }}
            >
              Why Choose Brain Vision?
            </Typography>
            <Grid container spacing={4}>
              {[
                {
                  icon: <VerifiedUser />,
                  title: "Accuracy",
                  description:
                    "Our AI model boasts a 99% accuracy rate in detecting brain tumors.",
                },
                {
                  icon: <Speed />,
                  title: "Speed",
                  description:
                    "Get results in minutes, not days. Fast analysis for quick decision making.",
                },
                {
                  icon: <Accessibility />,
                  title: "Accessibility",
                  description:
                    "Use our service from anywhere, anytime. All you need is an internet connection.",
                },
              ].map((feature, index) => (
                <Grid item xs={12} md={4} key={index}>
                  <Box
                    textAlign="center"
                    sx={{
                      transition: "transform 0.3s ease-in-out",
                      "&:hover": {
                        transform: "translateY(-5px)",
                      },
                    }}
                  >
                    <Box sx={{ color: "white", fontSize: 50, mb: 2 }}>
                      {feature.icon}
                    </Box>
                    <Typography variant="h6" gutterBottom color="white">
                      {feature.title}
                    </Typography>
                    <Typography variant="body2" color="white">
                      {feature.description}
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>

        <Box textAlign="center">
          <Typography
            variant="h4"
            gutterBottom
            sx={{
              color: "white",
              textShadow: "1px 1px 4px rgba(0,0,0,0.4)",
            }}
          >
            Ready to Get Started?
          </Typography>
          <Button
            component={RouterLink}
            to="/upload"
            variant="contained"
            size="large"
            endIcon={<ArrowForward />}
            sx={{
              mt: 3,
              px: 5,
              py: 2,
              fontSize: "1.2rem",
              bgcolor: "#ffffff",
              color: "#4A1C23",
              borderRadius: "30px",
              "&:hover": {
                bgcolor: "#f5f5f5",
              },
            }}
          >
            Upload Your Scan Now
          </Button>
        </Box>
      </Container>
    </Box>
  );
};

export default HomePage;
