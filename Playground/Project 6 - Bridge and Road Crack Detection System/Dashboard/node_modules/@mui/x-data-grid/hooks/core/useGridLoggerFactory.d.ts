import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommon } from "../../models/api/gridApiCommon.js";
import type { DataGridProcessedProps } from "../../models/props/DataGridProps.js";
export declare const useGridLoggerFactory: (apiRef: RefObject<GridPrivateApiCommon>, props: Pick<DataGridProcessedProps, "logger" | "logLevel">) => void;