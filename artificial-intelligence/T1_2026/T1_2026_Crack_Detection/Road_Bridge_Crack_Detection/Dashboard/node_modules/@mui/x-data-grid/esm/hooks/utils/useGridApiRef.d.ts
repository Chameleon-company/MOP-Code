import type { RefObject } from '@mui/x-internals/types';
import type { GridApiCommunity } from "../../models/api/gridApiCommunity.js";
/**
 * Hook that instantiate a [[GridApiRef]].
 */
export declare const useGridApiRef: () => RefObject<GridApiCommunity | null>;