import { Box } from '@mui/material';


export function SeverityBadge({ severity }: { severity: string }) {
  const s = String(severity).toLowerCase();
  const map: Record<string, { bg: string; border: string; text: string }> = {
    low:      { bg: 'rgba(99,153,34,0.15)',  border: 'rgba(99,153,34,0.35)',  text: '#a8d870' },
    medium:   { bg: 'rgba(186,117,23,0.15)', border: 'rgba(186,117,23,0.35)', text: '#d4a94e' },
    high:     { bg: 'rgba(226,75,74,0.15)',  border: 'rgba(226,75,74,0.35)',  text: '#f08080' },
    severe: { bg: 'rgba(226, 74, 74, 0.29)',   border: 'rgba(226,75,74,0.5)',   text: '#ff6b6b' },
  };
  const style = map[s] ?? { bg: 'rgba(255,255,255,0.06)', border: 'rgba(255,255,255,0.15)', text: 'rgba(255,255,255,0.5)' };

  return (
    <Box sx={{
      px: 1.5, py: 0.35,
      borderRadius: '5px',
      bgcolor: style.bg,
      border: `1px solid ${style.border}`,
      fontSize: '0.7rem', fontWeight: 700,
      color: style.text,
      textTransform: 'uppercase',
      letterSpacing: '0.06em',
    }}>
      {severity}
    </Box>
  );
}


export function SeverityBadgeLarge({ severity }: { severity: string }) {
  const s = String(severity).toLowerCase();
  const map: Record<string, { bg: string; border: string; text: string }> = {
    low: { bg: 'rgba(99,153,34,0.15)',  border: 'rgba(99,153,34,0.35)',  text: '#a8d870' },
    medium: { bg: 'rgba(186,117,23,0.15)', border: 'rgba(186,117,23,0.35)', text: '#d4a94e' },
    high: { bg: 'rgba(226,75,74,0.15)',  border: 'rgba(226,75,74,0.35)',  text: '#f08080' },
    severe: { bg: 'rgba(226,75,74,0.2)',   border: 'rgba(226,75,74,0.5)',   text: '#ff6b6b' },
  };
  const style = map[s] ?? { bg: 'rgba(255,255,255,0.06)', border: 'rgba(255,255,255,0.15)', text: 'rgba(255,255,255,0.5)' };

  return (
    <Box sx={{
      px: 3, py: 0.75,
      borderRadius: '8px',
      bgcolor: style.bg,
      border: `1px solid ${style.border}`,
      fontSize: '1rem', fontWeight: 700,
      color: style.text,
      textTransform: 'uppercase',
      letterSpacing: '0.06em',
    }}>
      {severity}
    </Box>
  );
}