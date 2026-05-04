import { useState } from 'react'
import type { crackReport } from '../Components/tableSelect'
import TableSelect from '../Components/tableSelect'
import Box from '@mui/material/Box'
import { TextField } from '@mui/material'
import TheDrawer from '../Components/drawer'

export function Home() {

  const [selectedRow, setSelectedRow] = useState<crackReport | null>(null)
  


  const handleRowSelect = (rowData: crackReport | null) => {
        setSelectedRow(rowData)
    }

  const formatReportData = (report: crackReport | null) => {
        if (!report) return ''
        
        return [
            `ID: ${report.id}`,
            `Image ID: ${report.imageid || 'N/A'}`,
            `Severity: ${report.severity}`,
            `Number of cracks: ${report.numcracks}`,
            `Largest crack area ratio: ${report.crackarearatio}`,
            `Estimated largest crack length ratio: ${report.estimatedcracklength}`,
            `Risk Assessment: ${report.riskassessment || 'N/A'}`,
            `Recommended repair: ${report.repairactions || 'N/A'}`,
            `Inspection Schedule: ${report.inspectionschedule || 'N/A'}`
        ].join('\n\n')
    }





  return (
    <>
      <div><TheDrawer></TheDrawer></div>
      <h1>Crack Database Dashboard</h1>
      <div style={{ marginTop: '20px', width: '90%', margin: '20px auto 0' }}><TableSelect onRowSelect={handleRowSelect}></TableSelect></div>

      {selectedRow && (
        <Box sx={{ marginTop: 3, width: '100%', maxWidth: '80%', marginX: 'auto' }}>
            <TextField
                fullWidth
                label="Report Details"
                multiline
                rows={12}
                value={formatReportData(selectedRow)}
                sx={{
                    '& .MuiInputBase-input': { color: 'white' },
                    '& .MuiInputLabel-root': { color: 'white' },
                }}
            />
            <h2 style={{marginTop: '20px'}}>Images:</h2>
            <details style={{marginBottom: '120px'}}>
              <summary>Click to expand</summary>
              <div style={{ display: 'flex', gap: '1rem' }}>
                <div>
                  <h3>Original Image:</h3>
                  <img src={selectedRow.imageurl} />
                  <p style={{ fontSize: "0.65rem" }}>URL: {selectedRow.imageurl}</p>
                </div>
                <div>
                  <h3>Binary Mask:</h3>
                  <img src={selectedRow.crackmaskurl} />
                  <p style={{ fontSize: "0.65rem" }}>URL: {selectedRow.crackmaskurl}</p>
                </div>
              </div>
            </details>
        </Box>
      )}
    </>
  )
  }

