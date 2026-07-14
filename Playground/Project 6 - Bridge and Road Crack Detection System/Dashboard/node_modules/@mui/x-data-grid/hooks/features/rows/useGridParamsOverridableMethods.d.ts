import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
export declare const useGridParamsOverridableMethods: (apiRef: RefObject<GridPrivateApiCommunity>) => {
  getCellValue: <V = any>(id: import("@mui/x-data-grid").GridRowId, field: string) => V;
  getRowValue: <V = any>(row: import("@mui/x-data-grid").GridRowModel, colDef: import("@mui/x-data-grid").GridColDef) => V;
  getRowFormattedValue: <V = any>(row: import("@mui/x-data-grid").GridRowModel, colDef: import("@mui/x-data-grid").GridColDef) => V;
};