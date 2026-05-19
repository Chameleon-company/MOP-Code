import TheDrawer from '../Components/drawer'
import MultipleFileUploader from '../Components/multipleUpload'
import { Box, Typography } from '@mui/material';






export function BatchUpload() {

    return (
        <Box sx={{ minHeight: '100vh', bgcolor: '#0f1117', color: '#e8eaf0' }}>
            <TheDrawer></TheDrawer>
            <Box sx={{
                borderBottom: '1px solid rgba(255,255,255,0.08)',
                px: 5, py: 3,
                display: 'flex', alignItems: 'flex-start', gap: 2,
            }}>
                <Box sx={{mx: 'auto'}}>
                    <Typography sx={{ fontSize: '1.35rem', fontWeight: 600, letterSpacing: '-0.01em', lineHeight: 1.2 }}>
                        Upload Images:
                    </Typography>
                    <Typography sx={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.4)', mt: 0.4, fontFamily: 'monospace' }}>
                        INPUT FORMAT: PNG / JPG / JPEG
                    </Typography>
                </Box>

            </Box>
            
            <div><MultipleFileUploader></MultipleFileUploader></div>


        </Box>


    )
}
