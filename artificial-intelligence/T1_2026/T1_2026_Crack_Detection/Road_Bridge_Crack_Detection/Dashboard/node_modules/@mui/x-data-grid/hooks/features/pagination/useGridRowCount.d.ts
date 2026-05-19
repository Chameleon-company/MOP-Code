import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { DataGridProcessedProps } from "../../../models/props/DataGridProps.js";
export declare const useGridRowCount: (apiRef: RefObject<GridPrivateApiCommunity>, props: Pick<DataGridProcessedProps, "rowCount" | "initialState" | "paginationMode" | "onRowCountChange">) => void;