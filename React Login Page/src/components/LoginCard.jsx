import { Box, Card, CardContent, TextField, Button, Typography } from "@mui/material";

export default function LoginCard() {
  return (
    <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh" bgcolor="#f5f5f5">
      <Card sx={{ width: 360, borderRadius: 3, boxShadow: 3 }}>
        <CardContent sx={{ p: 4 }}>
          <Typography variant="h5" align="center" fontWeight="bold" gutterBottom>
            Login
          </Typography>
          <Box component="form" sx={{ mt: 2, display: "flex", flexDirection: "column", gap: 2 }}>
            <TextField label="Email" variant="outlined" fullWidth size="small" />
            <TextField label="Password" type="password" variant="outlined" fullWidth size="small" />
            <Button variant="contained" fullWidth>
              Sign In
            </Button>
          </Box>
          <Typography variant="body2" align="center" sx={{ mt: 2, color: "text.secondary" }}>
            Don't have an account? <a href="#">Sign up</a>
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
}
