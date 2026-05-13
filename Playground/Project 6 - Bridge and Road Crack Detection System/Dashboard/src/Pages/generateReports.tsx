import TheDrawer from '../Components/drawer'
import { useState } from 'react'
import type { crackReport } from '../Components/tableSelect'
import NoReportTable from '../Components/noReportTable'
import Box from '@mui/material/Box'






export function GenerateReports() {
  const [refreshKey, setRefreshKey] = useState(0)
  const [selectedRows, setSelectedRows] = useState<crackReport[]>([])
  const [loading, setLoading] = useState(false)

  const handleRowSelect = (rows: crackReport[]) => {
    setSelectedRows(rows)
  }

  const handleButton = async () => {
    setLoading(true)
    console.log(selectedRows)
    const url = `${(import.meta as any).env.VITE_API_URL}/api/generateAiReport`
    let idArray: number[] = []
    selectedRows.forEach((row) => {
      idArray.push(row.id)
    })
    console.log(idArray)

    try {
      const result = await fetch(url, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ IDs: idArray})})

      if (result.ok) {
        console.log("AI Reports generated!")
      }
    }
    catch (error) {
      console.error(error)
    }
    setRefreshKey(prev => prev + 1)
    setLoading(false)

  }

  return (
    <>
    <Box sx={{ minHeight: '100vh', bgcolor: '#0f1117', color: '#e8eaf0' }}>
      <div><TheDrawer></TheDrawer></div>
      <h1>Cracks without AI generated reports:</h1>
      <div style={{ marginTop: '20px', width: '90%', margin: '20px auto 0' }}><NoReportTable key={refreshKey} onRowSelect={handleRowSelect} /></div>
      <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3, mt: 6 }}>
        <button 
            onClick={handleButton}
            className="submit"
            disabled={selectedRows.length === 0 || loading}
            style={{ fontSize: '1.2rem', padding: '12px 20px', borderRadius: '12px' }}
        >Generate AI reports for selected entries
        </button>
      </Box>
    </Box>
    </>
  )
  }