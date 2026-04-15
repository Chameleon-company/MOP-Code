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
    riskmanagement: string
    recommendedrepair: string
    nextaction: string
}

const columns: GridColDef[] = [
  { field: 'id', headerName: 'ID', width: 100 },
  { field: 'imageid', headerName: 'Image ID', width: 200 },
  { field: 'crackarearatio', headerName: 'Crack Area Ratio', width: 140 },
  { field: 'estimatedcracklength', headerName: 'Estimated Crack Length', width: 140 },
  { field: 'numcracks', headerName: 'Num Cracks', width: 100 },
  { field: 'severity', headerName: 'Severity', width: 130 },
  { field: 'riskmanagement', headerName: 'Risk Management', width: 330 },
  { field: 'recommendedrepair', headerName: 'Recommended Repair', width: 330 },
  { field: 'nextaction', headerName: 'Next Actions', width: 330 },
]

interface tableProps {
  onRowSelect: (rowData: crackReport | null) => void
}

const paginationModel = { page: 0, pageSize: 10 }



export default function TableSelect({ onRowSelect }: tableProps) {


    
  const [crackReport, setCrackReport] = useState<crackReport[]>([])
  useEffect(() => {
    getCrackReports()
  }, [])

  async function getCrackReports() {
    try {
      const url = `${(import.meta as any).env.VITE_API_URL}/api/getData`
      console.log(url)
      const res = await fetch(`${(import.meta as any).env.VITE_API_URL}/api/getData`)

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
    console.log("Selection model:", selectionModel)

    const selectedIds = Array.from(selectionModel.ids || [])
    console.log("Selected IDs:", selectedIds)

    if (selectedIds.length > 0) {
      const selectedId = selectedIds[0]
      console.log("Selected ID:", selectedId)
      const selectedReport = crackReport.find(s => s.id === selectedId)
      console.log("Found report:", selectedReport)
      onRowSelect(selectedReport || null)
    } else {
      onRowSelect(null)
    }



  }



  return (
    <Paper sx={{ height: 900, width: '100%' }}>
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
        disableMultipleRowSelection
        onRowSelectionModelChange={handleRowSelection}
        sx={{ border: 0, backgroundColor: '#cdcdcdff' }}
      />
    </Paper>
  )
}


  

export type {crackReport}