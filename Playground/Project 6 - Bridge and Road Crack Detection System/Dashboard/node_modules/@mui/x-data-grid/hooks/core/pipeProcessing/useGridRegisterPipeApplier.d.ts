import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommon } from "../../../models/api/gridApiCommon.js";
import type { GridPipeProcessorGroup } from "./gridPipeProcessingApi.js";
export declare const useGridRegisterPipeApplier: <PrivateApi extends GridPrivateApiCommon, G extends GridPipeProcessorGroup>(apiRef: RefObject<PrivateApi>, group: G, callback: () => void) => void;