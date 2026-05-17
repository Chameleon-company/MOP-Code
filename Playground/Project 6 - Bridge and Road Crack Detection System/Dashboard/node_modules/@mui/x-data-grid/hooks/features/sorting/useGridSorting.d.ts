import type { RefObject } from '@mui/x-internals/types';
import type { DataGridProcessedProps } from "../../../models/props/DataGridProps.js";
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { GridStateInitializer } from "../../utils/useGridInitializeState.js";
export declare const sortingStateInitializer: GridStateInitializer<Pick<DataGridProcessedProps, 'sortModel' | 'initialState' | 'disableMultipleColumnsSorting'>>;
/**
 * @requires useGridRows (event)
 * @requires useGridColumns (event)
 */
export declare const useGridSorting: (apiRef: RefObject<GridPrivateApiCommunity>, props: Pick<DataGridProcessedProps, "initialState" | "sortModel" | "onSortModelChange" | "sortingOrder" | "sortingMode" | "disableColumnSorting" | "disableMultipleColumnsSorting" | "multipleColumnsSortingMode" | "signature">) => void;