import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { DataGridProcessedProps } from "../../../models/props/DataGridProps.js";
import type { GridStateInitializer } from "../../utils/useGridInitializeState.js";
export declare const paginationStateInitializer: GridStateInitializer<Pick<DataGridProcessedProps, 'paginationModel' | 'rowCount' | 'initialState' | 'autoPageSize' | 'signature' | 'paginationMeta' | 'pagination' | 'paginationMode'>>;
/**
 * @requires useGridFilter (state)
 * @requires useGridDimensions (event) - can be after
 */
export declare const useGridPagination: (apiRef: RefObject<GridPrivateApiCommunity>, props: DataGridProcessedProps) => void;