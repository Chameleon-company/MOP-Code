import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { DataGridProcessedProps } from "../../../models/props/DataGridProps.js";
/**
 * @requires useGridColumns (state)
 * @requires useGridFilter (state)
 * @requires useGridSorting (state)
 * @requires useGridParamsApi (method)
 */
export declare const useGridPrintExport: (apiRef: RefObject<GridPrivateApiCommunity>, props: Pick<DataGridProcessedProps, "pagination" | "columnHeaderHeight" | "headerFilterHeight">) => void;