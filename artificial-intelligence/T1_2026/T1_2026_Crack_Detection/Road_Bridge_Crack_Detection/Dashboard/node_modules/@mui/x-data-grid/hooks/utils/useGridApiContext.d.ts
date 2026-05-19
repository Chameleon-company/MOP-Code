import type { RefObject } from '@mui/x-internals/types';
import type { GridApiCommon } from "../../models/api/gridApiCommon.js";
import type { GridApiCommunity } from "../../models/api/gridApiCommunity.js";
export declare function useGridApiContext<Api extends GridApiCommon = GridApiCommunity>(): RefObject<Api>;