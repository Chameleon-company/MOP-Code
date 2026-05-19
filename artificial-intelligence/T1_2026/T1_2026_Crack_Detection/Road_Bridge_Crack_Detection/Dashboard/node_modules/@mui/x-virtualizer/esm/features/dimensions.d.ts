import { Store } from '@mui/x-internals/store';
import { ColumnWithWidth, DimensionsState, RowId, RowsMetaState, Size } from "../models/index.js";
import type { BaseState, ParamsWithDefaults } from "../useVirtualizer.js";
export type DimensionsParams = {
  rowHeight: number;
  columnsTotalWidth?: number;
  leftPinnedWidth?: number;
  rightPinnedWidth?: number;
  topPinnedHeight?: number;
  bottomPinnedHeight?: number;
  autoHeight?: boolean;
  minimalContentHeight?: number | string;
  scrollbarSize?: number;
};
export declare const Dimensions: {
  initialize: typeof initializeState;
  use: typeof useDimensions;
  selectors: {
    rootSize: (state: BaseState) => Size;
    dimensions: (state: BaseState) => DimensionsState;
    rowHeight: (state: BaseState) => number;
    columnsTotalWidth: (state: BaseState) => number;
    contentHeight: (state: BaseState) => number;
    autoHeight: (state: BaseState) => boolean;
    minimalContentHeight: (state: BaseState) => string | number | undefined;
    rowsMeta: (state: BaseState) => RowsMetaState;
    rowPositions: (state: BaseState) => number[];
    columnPositions: (_: any, columns: ColumnWithWidth[]) => number[];
    needsHorizontalScrollbar: (state: BaseState) => boolean;
  };
};
export declare namespace Dimensions {
  type State = {
    rootSize: Size;
    dimensions: DimensionsState;
    rowsMeta: RowsMetaState;
    rowHeights: Map<any, any>;
  };
  type API = ReturnType<typeof useDimensions>;
}
declare function initializeState(params: ParamsWithDefaults): Dimensions.State;
declare function useDimensions(store: Store<BaseState>, params: ParamsWithDefaults, _api: {}): {
  updateDimensions: (firstUpdate?: boolean) => void;
  debouncedUpdateDimensions: (((firstUpdate?: boolean) => void) & import("@mui/x-internals/throttle").Cancelable) | undefined;
  rowsMeta: {
    getRowHeight: (rowId: RowId) => any;
    setLastMeasuredRowIndex: (index: number) => void;
    storeRowHeightMeasurement: (id: RowId, height: number) => void;
    hydrateRowsMeta: () => void;
    observeRowHeight: (element: Element, rowId: RowId) => () => void | undefined;
    rowHasAutoHeight: (id: RowId) => any;
    getRowHeightEntry: (rowId: RowId) => any;
    getLastMeasuredRowIndex: () => number;
    resetRowHeights: () => void;
  };
};
export declare function observeRootNode(node: Element | null, store: Store<BaseState>, setRootSize: (size: Size) => void): (() => void) | undefined;
export {};