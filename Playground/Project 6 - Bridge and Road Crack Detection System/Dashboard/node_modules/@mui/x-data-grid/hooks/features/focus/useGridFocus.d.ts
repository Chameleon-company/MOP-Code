import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { DataGridProcessedProps } from "../../../models/props/DataGridProps.js";
import type { GridStateInitializer } from "../../utils/useGridInitializeState.js";
export declare const focusStateInitializer: GridStateInitializer;
/**
 * @requires useGridParamsApi (method)
 * @requires useGridRows (method)
 * @requires useGridEditing (event)
 */
export declare const useGridFocus: (apiRef: RefObject<GridPrivateApiCommunity>, props: Pick<DataGridProcessedProps, "pagination" | "paginationMode">) => void;