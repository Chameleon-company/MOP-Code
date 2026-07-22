import type { RefObject } from '@mui/x-internals/types';
import type { DataGridProcessedProps } from "../../../models/props/DataGridProps.js";
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { GridStateInitializer } from "../../utils/useGridInitializeState.js";
export declare const rowSelectionStateInitializer: GridStateInitializer<Pick<DataGridProcessedProps, 'rowSelectionModel' | 'rowSelection'>>;
/**
 * @requires useGridRows (state, method) - can be after
 * @requires useGridParamsApi (method) - can be after
 * @requires useGridFocus (state) - can be after
 * @requires useGridKeyboardNavigation (`cellKeyDown` event must first be consumed by it)
 */
export declare const useGridRowSelection: (apiRef: RefObject<GridPrivateApiCommunity>, props: Pick<DataGridProcessedProps, "checkboxSelection" | "rowSelectionModel" | "onRowSelectionModelChange" | "disableMultipleRowSelection" | "disableRowSelectionOnClick" | "disableRowSelectionExcludeModel" | "isRowSelectable" | "checkboxSelectionVisibleOnly" | "pagination" | "paginationMode" | "filterMode" | "classes" | "keepNonExistentRowsSelected" | "rowSelection" | "rowSelectionPropagation" | "signature">) => void;