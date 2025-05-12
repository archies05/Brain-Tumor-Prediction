import React from 'react';
import { Box, Typography, Stack } from '@mui/material';

const Label = () => {
  const labels = [
    { color: '#2020B9', label: 'Necrotic core' },
    { color: '#66FF66', label: 'Edema' },
    { color: '#EAAD51', label: 'Enhancing tumor' },
  ];

  return (
    <Box
      sx={{
        border: '1px solid gray',
        p: 2,
        backgroundColor: '#f5f5f5',
        width: 220,
        borderRadius: 1,
      }}
    >
      <Typography variant="subtitle1" fontWeight="bold" mb={1}>
        Tumor Regions
      </Typography>
      <Stack spacing={1}>
        {labels.map((item) => (
          <Stack key={item.label} direction="row" alignItems="center" spacing={1}>
            <Box
              sx={{
                width: 20,
                height: 20,
                backgroundColor: item.color,
                borderRadius: '4px',
                border: '1px solid #000',
              }}
            />
            <Typography variant="body2">{item.label}</Typography>
          </Stack>
        ))}
      </Stack>
    </Box>
  );
};

export default Label;
