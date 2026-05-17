import type { GridStateCommunity } from "../../../models/gridStateCommunity.js";
import type { GridRowId } from "../../../models/gridRows.js";
export declare const gridRowReorderStateSelector: import("@mui/x-data-grid").OutputSelector<GridStateCommunity, unknown, import("./gridRowReorderInterfaces.js").GridRowReorderState>;
export declare const gridIsRowDragActiveSelector: (args_0: import("react").RefObject<{
  state: GridStateCommunity;
} | null>) => boolean;
export declare const gridRowDropTargetSelector: (args_0: import("react").RefObject<{
  state: GridStateCommunity;
} | null>) => {
  rowId: GridRowId;
  position: import("../../../internals/index.js").RowReorderDropPosition;
} | {
  rowId: null;
  position: null;
};
export declare const gridRowDropTargetRowIdSelector: (args_0: import("react").RefObject<{
  state: GridStateCommunity;
} | null>) => GridRowId | null;
export declare const gridRowDropPositionSelector: (args_0: import("react").RefObject<{
  state: GridStateCommunity;
} | null>, rowId: GridRowId) => import("../../../internals/index.js").RowReorderDropPosition | null;
export declare const gridDraggedRowIdSelector: (args_0: import("react").RefObject<{
  state: GridStateCommunity;
} | null>) => GridRowId | null;