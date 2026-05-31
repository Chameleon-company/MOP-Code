import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommon } from "../../models/api/gridApiCommon.js";
import type { GridStateProps } from "../../models/gridStateCommunity.js";
import type { GridStateInitializer } from "../utils/useGridInitializeState.js";
export declare const propsStateInitializer: GridStateInitializer<GridStateProps>;
export declare const useGridProps: <PrivateApi extends GridPrivateApiCommon>(apiRef: RefObject<PrivateApi>, props: GridStateProps) => void;