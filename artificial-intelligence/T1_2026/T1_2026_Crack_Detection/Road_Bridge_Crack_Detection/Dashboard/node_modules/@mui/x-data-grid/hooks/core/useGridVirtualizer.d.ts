import * as React from 'react';
import { Dimensions, LayoutDataGridLegacy, Virtualization } from '@mui/x-virtualizer';
/**
 * Virtualizer setup
 */
export declare function useGridVirtualizer(): {
  store: import("@mui/x-internals/store").Store<Dimensions.State & Virtualization.State<LayoutDataGridLegacy> & import("@mui/x-virtualizer").Colspan.State & import("@mui/x-virtualizer").Rowspan.State>;
  api: {
    updateDimensions: (firstUpdate?: boolean) => void;
    debouncedUpdateDimensions: (((firstUpdate?: boolean) => void) & import("@mui/x-internals/throttle").Cancelable) | undefined;
    rowsMeta: {
      getRowHeight: (rowId: import("@mui/x-virtualizer/models").RowId) => any;
      setLastMeasuredRowIndex: (index: number) => void;
      storeRowHeightMeasurement: (id: import("@mui/x-virtualizer/models").RowId, height: number) => void;
      hydrateRowsMeta: () => void;
      observeRowHeight: (element: Element, rowId: import("@mui/x-virtualizer/models").RowId) => () => void | undefined;
      rowHasAutoHeight: (id: import("@mui/x-virtualizer/models").RowId) => any;
      getRowHeightEntry: (rowId: import("@mui/x-virtualizer/models").RowId) => any;
      getLastMeasuredRowIndex: () => number;
      resetRowHeights: () => void;
    };
  } & {
    getCellColSpanInfo: (rowId: import("@mui/x-virtualizer/models").RowId, columnIndex: import("@mui/x-internals/types").integer) => import("@mui/x-virtualizer/models").CellColSpanInfo;
    calculateColSpan: (rowId: import("@mui/x-virtualizer/models").RowId, minFirstColumn: import("@mui/x-internals/types").integer, maxLastColumn: import("@mui/x-internals/types").integer, columns: import("@mui/x-virtualizer/models").ColumnWithWidth[]) => void;
    getHiddenCellsOrigin: () => Record<import("@mui/x-virtualizer/models").RowId, Record<number, number>>;
    getters: any;
    setPanels: React.Dispatch<React.SetStateAction<Readonly<Map<any, React.ReactNode>>>>;
    forceUpdateRenderContext: () => void;
    scheduleUpdateRenderContext: () => void;
  } & {
    resetColSpan: () => void;
    getCellColSpanInfo: (rowId: import("@mui/x-virtualizer/models").RowId, columnIndex: import("@mui/x-internals/types").integer) => import("@mui/x-virtualizer/models").CellColSpanInfo | undefined;
    calculateColSpan: (rowId: import("@mui/x-virtualizer/models").RowId, minFirstColumn: import("@mui/x-internals/types").integer, maxLastColumn: import("@mui/x-internals/types").integer, columns: import("@mui/x-virtualizer/models").ColumnWithWidth[]) => void;
  } & {
    getHiddenCellsOrigin: () => Record<number, Record<number, number>>;
  } & {
    getViewportPageSize: () => number;
  };
};