import { DataGrid } from '@mui/x-data-grid'
import type { GridColDef } from '@mui/x-data-grid'
import Paper from '@mui/material/Paper'
import { useEffect, useState } from "react"






type crackReport = {
    id: number
    crackarearatio: number
    estimatedcracklength: number
    numcracks: number
    severity: number
    imageid: string
    riskassessment: string
    repairactions: string
    inspectionschedule: string
}

const columns: GridColDef[] = [
  { field: 'id', headerName: 'ID', width: 100 },
  { field: 'imageid', headerName: 'Image ID', width: 200 },
  { field: 'severity', headerName: 'Severity', width: 130 },
  { field: 'numcracks', headerName: 'Num Cracks', width: 180 },
  { field: 'crackarearatio', headerName: 'Crack Area Ratio', width: 140 },
  { field: 'estimatedcracklength', headerName: 'Estimated Crack Length', width: 250 },
  { field: 'damagelevel', headerName: 'Damage Level', width: 150},
  { field: 'reportstatus', headerName: 'Report Status', width: 150}
]

interface tableProps {
  onRowSelect: (reports: crackReport[]) => void
}

const paginationModel = { page: 0, pageSize: 50 }



export default function NoReportTable({ onRowSelect }: tableProps) {


    
  const [crackReport, setCrackReport] = useState<crackReport[]>([])
  useEffect(() => {
    getCrackReports()
  }, [])

  async function getCrackReports() {
    try {
      const url = `${(import.meta as any).env.VITE_API_URL}/api/getNoReportData`
      console.log(url)
      const res = await fetch(`${(import.meta as any).env.VITE_API_URL}/api/getNoReportData`)

      if(!res.ok) {
        throw new Error("failed to fetch crack report")
      }

      const data = await res.json()
      setCrackReport(data.Data ?? [])

    }
    catch (err) {
      console.error(err)
      setCrackReport([])
    }
  }

  const handleRowSelection = (selectionModel: any) => {

    const selectedIds = Array.from(selectionModel.ids || [])

    if (selectedIds.length > 0) {

      const selectedReports = crackReport.filter(s => selectedIds.includes(s.id))

      onRowSelect(selectedReports)
    } else {
      onRowSelect([])
    }



  }



  return (
    <Paper sx={{ height: 650, width: '100%' }}>
      <DataGrid
        rows={crackReport}
        columns={columns}
        getRowId={(row) => row.id}
        initialState={{ 
          pagination: { paginationModel },
          sorting: {
            sortModel: [{ field: 'id', sort: 'asc' }]
          }
        }}
        pageSizeOptions={[50, 100, 200, 500]}
        checkboxSelection
        onRowSelectionModelChange={handleRowSelection}
        sx={{ border: 0, backgroundColor: '#cdcdcdff' }}
      />
    </Paper>
  )
}


  

export type {crackReport}