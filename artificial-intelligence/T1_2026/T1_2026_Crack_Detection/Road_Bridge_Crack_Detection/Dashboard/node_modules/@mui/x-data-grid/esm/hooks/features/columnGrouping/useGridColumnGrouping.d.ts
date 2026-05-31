import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { DataGridProcessedProps } from "../../../models/props/DataGridProps.js";
import type { GridStateInitializer } from "../../utils/useGridInitializeState.js";
export declare const columnGroupsStateInitializer: GridStateInitializer<Pick<DataGridProcessedProps, 'columnGroupingModel'>>;
/**
 * @requires useGridColumns (method, event)
 * @requires useGridParamsApi (method)
 */
export declare const useGridColumnGrouping: (apiRef: RefObject<GridPrivateApiCommunity>, props: Pick<DataGridProcessedProps, "columnGroupingModel">) => void;