import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { DataGridProcessedProps } from "../../../models/props/DataGridProps.js";
export declare const useGridCellEditing: (apiRef: RefObject<GridPrivateApiCommunity>, props: Pick<DataGridProcessedProps, "editMode" | "processRowUpdate" | "onCellEditStart" | "onCellEditStop" | "cellModesModel" | "onCellModesModelChange" | "onProcessRowUpdateError" | "signature" | "dataSource">) => void;