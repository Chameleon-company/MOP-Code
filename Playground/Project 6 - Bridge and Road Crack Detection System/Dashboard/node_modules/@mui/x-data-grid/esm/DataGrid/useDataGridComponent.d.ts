import type { RefObject } from '@mui/x-internals/types';
import type { DataGridProcessedProps } from "../models/props/DataGridProps.js";
import type { GridPrivateApiCommunity } from "../models/api/gridApiCommunity.js";
import type { GridConfiguration } from "../models/configuration/gridConfiguration.js";
export declare const useDataGridComponent: (apiRef: RefObject<GridPrivateApiCommunity>, props: DataGridProcessedProps, configuration: GridConfiguration) => void;