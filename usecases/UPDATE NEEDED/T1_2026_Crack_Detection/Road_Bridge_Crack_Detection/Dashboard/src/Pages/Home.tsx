import { useState } from 'react'
import type { crackReport } from '../Components/tableSelect'
import TableSelect from '../Components/tableSelect'
import { Box, Typography } from '@mui/material';
import {SeverityBadgeLarge} from '../Components/severityBadge'
import TheDrawer from '../Components/drawer'

export function Home() {
  const [selectedRow, setSelectedRow] = useState<crackReport | null>(null)
  
  const handleRowSelect = (rowData: crackReport | null) => {
    setSelectedRow(rowData)
  }

  return (
    <>
    <Box sx={{ minHeight: '100vh', bgcolor: '#0f1117', color: '#e8eaf0' }}>
      <div><TheDrawer></TheDrawer></div>
      <h1>Crack Database Dashboard</h1>
      <Typography sx={{ fontSize: '1.2rem', color: 'rgba(255, 255, 255, 0.53)', mt: 0.4, fontFamily: 'monospace', marginBottom: 2, letterSpacing: '-0.03em' }}>
        Please select a row to view infomation and images
      </Typography>

      <div style={{ marginTop: '20px', width: '90%', margin: '20px auto 0' }}><TableSelect onRowSelect={handleRowSelect}></TableSelect></div>

      {selectedRow && (

        <Box sx={{ marginTop: 3, width: '100%', maxWidth: '80%', marginX: 'auto' }}>
          <Box sx={{ minHeight: '20vh', bgcolor: '#141720', color: '#e8eaf0', borderRadius: '12px', marginTop: 2 }}>

            <Box sx={{maxWidth: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', mb: '40px', mt: '20px'}}>
              <Typography sx={{ fontSize: '1rem', color: 'rgba(255, 255, 255, 0.76)', mb: 0.4, textTransform: 'uppercase', letterSpacing: '0.05em' ,marginRight: '150px', mt: '20px' }}>
                  Report for: {selectedRow.imageid}
              </Typography>
              <Typography sx={{ fontSize: '1rem', color: 'rgba(255, 255, 255, 0.76)', mb: 0.4, textTransform: 'uppercase', letterSpacing: '0.05em', marginRight: '10px', mt: '20px' }}>
                  Severity:
              </Typography>
              <Box sx={{marginTop: '20px'}}><SeverityBadgeLarge severity={selectedRow.severity} /></Box>

            </Box>
            <Box sx={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0}}>
              {[
              ['ID', selectedRow.id],
              ['Damage level', selectedRow.damagelevel],
              ['Crack area ratio', selectedRow.crackarearatio],
              ['Est. crack length', selectedRow.estimatedcracklength],
              ['Crack regions', selectedRow.numcracks],
              ['Report status', selectedRow.reportstatus],
              ['Inspection schedule', selectedRow.inspectionschedule],
              ['Risk assessment', selectedRow.riskassessment],
              ['Repair actions', selectedRow.repairactions],
              ].map(([label, value], i) => (
              <Box key={label as string} sx={{
                  px: 2.5, py: 1.5,
                  borderBottom: i < 6 ? '1px solid rgba(255,255,255,0.05)' : 'none',
                  borderRight: i % 2 === 0 ? '1px solid rgba(255,255,255,0.05)' : 'none',
              }}>
                  <Typography sx={{ fontSize: '0.68rem', color: 'rgba(255,255,255,0.28)', mb: 0.4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  {label}
                  </Typography>
                  <Typography sx={{ fontSize: '0.85rem', color: 'rgba(255,255,255,0.8)', fontFamily: 'monospace' }}>
                  {String(value)}
                  </Typography>
              </Box>
              ))}
            </Box>
          </Box>
            
            <h2 style={{marginTop: '20px'}}>Images:</h2>
            <details style={{marginBottom: '120px'}}>
              <summary>
                Click to show images
              </summary>
              <div style={{ display: 'flex', gap: '1rem' }}>
                <div>
                  <Typography sx={{ fontSize: '1.2rem', color: 'rgba(255, 255, 255, 0.53)', mt: 0.4, fontFamily: 'monospace', marginBottom: 2 }}>
                        Original Image:
                  </Typography>
                  <img src={selectedRow.imageurl} />
                  <p style={{ fontSize: "0.65rem" }}>URL: {selectedRow.imageurl}</p>
                </div>
                <div>
                  <Typography sx={{ fontSize: '1.2rem', color: 'rgba(255, 255, 255, 0.53)', mt: 0.4, fontFamily: 'monospace', marginBottom: 2 }}>
                        Binary Mask:
                  </Typography>
                  <img src={selectedRow.crackmaskurl} />
                  <p style={{ fontSize: "0.65rem" }}>URL: {selectedRow.crackmaskurl}</p>
                </div>
              </div>

              <div>                                          {/* NEW */}
                <h3>Crack Overlay:</h3>
                {selectedRow.overlayurl
                  ? <img src={selectedRow.overlayurl} />
                  : <p style={{ color: 'grey' }}>No overlay available</p>
                }
                <p style={{ fontSize: "0.65rem" }}>URL: {selectedRow.overlayurl}</p>
              </div>
          </details>
        </Box>
      )}
    </Box>
    </>
  )
}