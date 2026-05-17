import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { DataGridProcessedProps } from "../../../models/props/DataGridProps.js";
import type { GridStateInitializer } from "../../utils/useGridInitializeState.js";
export declare const columnsStateInitializer: GridStateInitializer<Pick<DataGridProcessedProps, 'columnVisibilityModel' | 'initialState' | 'columns'>>;
/**
 * @requires useGridParamsApi (method)
 * @requires useGridDimensions (method, event) - can be after
 * TODO: Impossible priority - useGridParamsApi also needs to be after useGridColumns
 */
export declare function useGridColumns(apiRef: RefObject<GridPrivateApiCommunity>, props: Pick<DataGridProcessedProps, 'initialState' | 'columns' | 'columnVisibilityModel' | 'onColumnVisibilityModelChange' | 'slots' | 'slotProps' | 'disableColumnSelector' | 'signature'>): void;