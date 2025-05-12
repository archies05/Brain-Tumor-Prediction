import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { Box, Container, Grid, Typography, Link, IconButton } from '@mui/material';
import { Facebook, Twitter, LinkedIn, Instagram } from '@mui/icons-material';

const Footer: React.FC = () => {
  return (
    <Box 
      component="footer" 
      sx={{ 
        // background: 'linear-gradient(90deg, #000 0%, #6a11cb 100%)',
        color: 'white', 
        py: 6 
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          {/* Brand and Description */}
          <Grid item xs={12} sm={4}>
            <Typography 
              variant="h5" 
              gutterBottom 
              sx={{ fontWeight: 'bold', color: 'white' }}
            >
              Brain Vision
            </Typography>
            <Typography variant="body2" sx={{ lineHeight: 1.8 }}>
              Revolutionizing healthcare with AI-powered brain tumor detection. Accurate, fast, and accessible solutions for everyone.
            </Typography>
          </Grid>

          {/* Quick Links */}
          <Grid item xs={12} sm={4}>
            <Typography 
              variant="h6" 
              gutterBottom 
              sx={{ color: 'white', fontWeight: 'bold' }}
            >
              Quick Links
            </Typography>
            <Box>
              {['Home', 'Upload Scan'].map((link, index) => (
                <Link
                  key={index}
                  component={RouterLink}
                  to={`/${link.toLowerCase().replace(/\s+/g, '')}`}
                  color="inherit"
                  underline="none"
                  display="block"
                  sx={{ mb: 1, '&:hover': { textDecoration: 'underline' } }}
                >
                  {link}
                </Link>
              ))}
            </Box>
          </Grid>

          <Grid item xs={12} sm={4}>
            <Typography 
              variant="h6" 
              gutterBottom 
              sx={{ color: 'white', fontWeight: 'bold' }}
            >
              Contact Us
            </Typography>
            <Typography variant="body2" sx={{ lineHeight: 1.8 }}>
              Email: info@brainscanai.com
            </Typography>
            <Typography variant="body2" sx={{ lineHeight: 1.8 }}>
              Phone: +91 - 9876543210
            </Typography>

            <Box mt={2}>
              {[
                { icon: <Facebook />, url: '#' },
                { icon: <Twitter />, url: '#' },
                { icon: <LinkedIn />, url: '#' },
                { icon: <Instagram />, url: '#' },
              ].map((social, index) => (
                <IconButton 
                  key={index}
                  href={social.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  sx={{ color: 'white', '&:hover': { color: 'grey.300' } }}
                >
                  {social.icon}
                </IconButton>
              ))}
            </Box>
          </Grid>
        </Grid>

        <Box mt={5} textAlign="center" borderTop={1} borderColor="rgba(255, 255, 255, 0.2)" pt={3}>
          <Typography 
            variant="body2" 
            sx={{ color: 'rgba(255, 255, 255, 0.8)' }}
          >
            © {new Date().getFullYear()} Brain Vision. All rights reserved.
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer;
