import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { GridColDef } from "../../../models/colDef/gridColDef.js";
import type { GridColumnMenuSlotProps } from "./columnMenuInterfaces.js";
export interface GetColumnMenuItemKeysParams {
  apiRef: RefObject<GridPrivateApiCommunity>;
  colDef: GridColDef;
  defaultSlots: Record<string, any>;
  defaultSlotProps: Record<string, GridColumnMenuSlotProps>;
  slots?: Record<string, any>;
  slotProps?: Record<string, GridColumnMenuSlotProps>;
}
/**
 * Returns the list of column menu item keys (sorted by `displayOrder`) that should be rendered for a given column.
 * This is shared between the column header (to know if menu is empty) and the menu itself (to render items).
 */
export declare function getColumnMenuItemKeys(params: GetColumnMenuItemKeysParams): string[];