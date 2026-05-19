import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { GridStateInitializer } from "../../utils/useGridInitializeState.js";
export declare const columnMenuStateInitializer: GridStateInitializer;
/**
 * @requires useGridColumnResize (event)
 * @requires useGridInfiniteLoader (event)
 */
export declare const useGridColumnMenu: (apiRef: RefObject<GridPrivateApiCommunity>) => void;