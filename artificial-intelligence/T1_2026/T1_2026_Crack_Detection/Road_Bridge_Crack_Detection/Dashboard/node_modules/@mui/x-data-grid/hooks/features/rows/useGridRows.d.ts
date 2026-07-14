import type { RefObject } from '@mui/x-internals/types';
import type { DataGridProcessedProps } from "../../../models/props/DataGridProps.js";
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { GridStateInitializer } from "../../utils/useGridInitializeState.js";
import type { GridConfiguration } from "../../../models/configuration/gridConfiguration.js";
export declare const rowsStateInitializer: GridStateInitializer<Pick<DataGridProcessedProps, 'dataSource' | 'rows' | 'rowCount' | 'getRowId' | 'loading'>>;
export declare const useGridRows: (apiRef: RefObject<GridPrivateApiCommunity>, props: Pick<DataGridProcessedProps, "rows" | "getRowId" | "rowCount" | "throttleRowsMs" | "signature" | "pagination" | "paginationMode" | "loading" | "dataSource" | "processRowUpdate" | "onProcessRowUpdateError">, configuration: GridConfiguration) => void;