import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { DataGridProcessedProps } from "../../../models/props/DataGridProps.js";
export declare const useGridRowEditing: (apiRef: RefObject<GridPrivateApiCommunity>, props: Pick<DataGridProcessedProps, "editMode" | "processRowUpdate" | "onRowEditStart" | "onRowEditStop" | "onProcessRowUpdateError" | "rowModesModel" | "onRowModesModelChange" | "signature" | "dataSource">) => void;