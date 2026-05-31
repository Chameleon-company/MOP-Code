import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { DataGridProcessedProps } from "../../../models/props/DataGridProps.js";
import type { GridConfiguration } from "../../../models/configuration/gridConfiguration.js";
import type { GridStateInitializer } from "../../utils/useGridInitializeState.js";
export declare const editingStateInitializer: GridStateInitializer;
export declare const useGridEditing: (apiRef: RefObject<GridPrivateApiCommunity>, props: Pick<DataGridProcessedProps, "isCellEditable" | "editMode" | "processRowUpdate" | "dataSource" | "onDataSourceError">, configuration: GridConfiguration) => void;