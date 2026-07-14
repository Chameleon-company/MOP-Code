import * as React from 'react';
import { integer } from '@mui/x-internals/types';
import { Store } from '@mui/x-internals/store';
import { Colspan } from "./features/colspan.js";
import { Dimensions } from "./features/dimensions.js";
import { Rowspan } from "./features/rowspan.js";
import { Virtualization } from "./features/virtualization/index.js";
import type { Layout } from "./features/virtualization/layout.js";
import type { HeightEntry, RowSpacing } from "./models/dimensions.js";
import type { ColspanParams } from "./features/colspan.js";
import type { DimensionsParams } from "./features/dimensions.js";
import type { VirtualizationParams } from "./features/virtualization/index.js";
import { ColumnWithWidth, FocusedCell, Size, PinnedRows, PinnedColumns, RenderContext, Row, RowEntry } from "./models/index.js";
export type Virtualizer = ReturnType<typeof useVirtualizer>;
export type VirtualScrollerCompat<L extends Layout = Layout> = Virtualization.State<L>['getters'];
export type BaseState<L extends Layout = Layout> = Virtualization.State<L> & Dimensions.State;
export type VirtualizerParams<L extends Layout = Layout> = {
  layout: L;
  dimensions: DimensionsParams;
  virtualization: VirtualizationParams;
  colspan?: ColspanParams;
  initialState?: {
    scroll?: {
      top: number;
      left: number;
    };
    rowSpanning?: Rowspan.State['rowSpanning'];
    virtualization?: Partial<Virtualization.State<L>['virtualization']>;
  };
  /** current page rows */
  rows: RowEntry[];
  /** current page range */
  range: {
    firstRowIndex: integer;
    lastRowIndex: integer;
  } | null;
  rowCount: integer;
  columns?: ColumnWithWidth[];
  pinnedRows?: PinnedRows;
  pinnedColumns?: PinnedColumns;
  disableHorizontalScroll?: boolean;
  disableVerticalScroll?: boolean;
  getRowHeight?: (row: RowEntry) => number | null | undefined | 'auto';
  /**
   * Function that returns the estimated height for a row.
   * Only works if dynamic row height is used.
   * Once the row height is measured this value is discarded.
   * @param rowEntry
   * @returns The estimated row height value. If `null` or `undefined` then the default row height, based on the density, is applied.
   */
  getEstimatedRowHeight?: (rowEntry: RowEntry) => number | null;
  /**
   * Function that allows to specify the spacing between rows.
   * @param rowEntry
   * @returns The row spacing values.
   */
  getRowSpacing?: (rowEntry: RowEntry) => RowSpacing;
  /** Update the row height values before they're used.
   * Used to add detail panel heights.
   * @param entry
   * @param rowEntry
   */
  applyRowHeight?: (entry: HeightEntry, rowEntry: RowEntry) => void;
  virtualizeColumnsWithAutoRowHeight?: boolean;
  resizeThrottleMs?: number;
  onResize?: (lastSize: Size) => void;
  onWheel?: (event: React.WheelEvent) => void;
  onTouchMove?: (event: React.TouchEvent) => void;
  onRenderContextChange?: (c: RenderContext) => void;
  onScrollChange?: (scrollPosition: {
    top: number;
    left: number;
  }, nextRenderContext: RenderContext) => void;
  focusedVirtualCell?: () => FocusedCell | null;
  scrollReset?: any;
  renderRow: (params: {
    id: any;
    model: Row;
    rowIndex: number;
    offsetLeft: number;
    columnsTotalWidth: number;
    baseRowHeight: number | 'auto';
    firstColumnIndex: number;
    lastColumnIndex: number;
    focusedColumnIndex: number | undefined;
    isFirstVisible: boolean;
    isLastVisible: boolean;
    isVirtualFocusRow: boolean;
    showBottomBorder: boolean;
  }) => React.ReactElement;
  renderInfiniteLoadingTrigger?: (id: any) => React.ReactElement;
};
type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;
export type ParamsWithDefaults = RequiredFields<VirtualizerParams, 'resizeThrottleMs' | 'columns'> & {
  dimensions: RequiredFields<VirtualizerParams['dimensions'], 'columnsTotalWidth' | 'leftPinnedWidth' | 'rightPinnedWidth' | 'topPinnedHeight' | 'bottomPinnedHeight' | 'autoHeight'>;
  virtualization: RequiredFields<VirtualizerParams['virtualization'], 'isRtl' | 'rowBufferPx' | 'columnBufferPx'>;
};
export declare const useVirtualizer: <L extends Layout = Layout>(params: VirtualizerParams<L>) => {
  store: Store<Dimensions.State & Virtualization.State<L> & Colspan.State & Rowspan.State>;
  api: {
    updateDimensions: (firstUpdate?: boolean) => void;
    debouncedUpdateDimensions: (((firstUpdate?: boolean) => void) & import("@mui/x-internals/throttle").Cancelable) | undefined;
    rowsMeta: {
      getRowHeight: (rowId: import("./models/index.js").RowId) => any;
      setLastMeasuredRowIndex: (index: number) => void;
      storeRowHeightMeasurement: (id: import("./models/index.js").RowId, height: number) => void;
      hydrateRowsMeta: () => void;
      observeRowHeight: (element: Element, rowId: import("./models/index.js").RowId) => () => void | undefined;
      rowHasAutoHeight: (id: import("./models/index.js").RowId) => any;
      getRowHeightEntry: (rowId: import("./models/index.js").RowId) => any;
      getLastMeasuredRowIndex: () => number;
      resetRowHeights: () => void;
    };
  } & {
    getCellColSpanInfo: (rowId: import("./models/index.js").RowId, columnIndex: integer) => import("./models/index.js").CellColSpanInfo;
    calculateColSpan: (rowId: import("./models/index.js").RowId, minFirstColumn: integer, maxLastColumn: integer, columns: ColumnWithWidth[]) => void;
    getHiddenCellsOrigin: () => Record<import("./models/index.js").RowId, Record<number, number>>;
    getters: any;
    setPanels: React.Dispatch<React.SetStateAction<Readonly<Map<any, React.ReactNode>>>>;
    forceUpdateRenderContext: () => void;
    scheduleUpdateRenderContext: () => void;
  } & {
    resetColSpan: () => void;
    getCellColSpanInfo: (rowId: import("./models/index.js").RowId, columnIndex: integer) => import("./models/index.js").CellColSpanInfo | undefined;
    calculateColSpan: (rowId: import("./models/index.js").RowId, minFirstColumn: integer, maxLastColumn: integer, columns: ColumnWithWidth[]) => void;
  } & {
    getHiddenCellsOrigin: () => Record<number, Record<number, number>>;
  } & {
    getViewportPageSize: () => number;
  };
};
export {};