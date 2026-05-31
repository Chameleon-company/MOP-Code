import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommunity } from "../api/gridApiCommunity.js";
import type { DataGridProcessedProps } from "../props/DataGridProps.js";
import type { GridEditingApi } from "../api/gridEditingApi.js";
/**
 * Get the cell editable condition function
 * @param {Object} params The cell parameters
 * @param {Object} params.rowNode The row node
 * @param {Object} params.colDef The column definition
 * @param {any} params.value The cell value
 * @returns {boolean} Whether the cell is editable
 */
export type CellEditableConditionFn = (params: Parameters<GridEditingApi['isCellEditable']>[0]) => boolean;
/**
 * Cell editable configuration interface for internal hooks
 */
export interface GridCellEditableInternalHook<Api = GridPrivateApiCommunity, Props = DataGridProcessedProps> {
  useIsCellEditable: (apiRef: RefObject<Api>, props: Props) => CellEditableConditionFn;
}