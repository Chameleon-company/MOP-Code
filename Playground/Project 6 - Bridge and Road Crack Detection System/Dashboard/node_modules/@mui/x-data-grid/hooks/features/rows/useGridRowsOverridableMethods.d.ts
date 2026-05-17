import type { RefObject } from '@mui/x-internals/types';
import type { GridRowId } from "../../../models/gridRows.js";
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
export declare const useGridRowsOverridableMethods: (apiRef: RefObject<GridPrivateApiCommunity>) => {
  setRowIndex: (rowId: GridRowId, targetIndex: number) => void;
  setRowPosition: (sourceRowId: GridRowId, targetRowId: GridRowId, position: import("../../../internals/index.js").RowReorderDropPosition) => void | Promise<void>;
};