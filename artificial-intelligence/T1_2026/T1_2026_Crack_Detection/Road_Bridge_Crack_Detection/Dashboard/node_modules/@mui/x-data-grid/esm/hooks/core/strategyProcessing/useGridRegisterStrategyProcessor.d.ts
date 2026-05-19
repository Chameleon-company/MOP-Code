import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommon } from "../../../models/api/gridApiCommon.js";
import type { GridStrategyProcessorName, GridStrategyProcessor } from "./gridStrategyProcessingApi.js";
export declare const useGridRegisterStrategyProcessor: <Api extends GridPrivateApiCommon, G extends GridStrategyProcessorName>(apiRef: RefObject<Api>, strategyName: string, group: G, processor: GridStrategyProcessor<G>) => void;