import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { DataGridProcessedProps } from "../../../models/props/DataGridProps.js";
import type { GridStateInitializer } from "../../utils/useGridInitializeState.js";
export declare const densityStateInitializer: GridStateInitializer<Pick<DataGridProcessedProps, 'initialState' | 'density'>>;
export declare const useGridDensity: (apiRef: RefObject<GridPrivateApiCommunity>, props: Pick<DataGridProcessedProps, "density" | "onDensityChange" | "initialState">) => void;