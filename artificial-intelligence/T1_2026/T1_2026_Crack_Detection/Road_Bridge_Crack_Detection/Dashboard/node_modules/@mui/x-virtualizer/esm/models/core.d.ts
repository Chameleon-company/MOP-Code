export type Size = {
  width: number;
  height: number;
};
export declare const Size: {
  EMPTY: {
    width: number;
    height: number;
  };
  equals: (a: Size, b: Size) => boolean;
};
export type Row = {
  [key: string | symbol]: any;
};
export type Column = {
  [key: string | symbol]: any;
};
export type ColumnWithWidth = {
  computedWidth: number;
} & Column;
export type RowId = any;
export type RowEntry = {
  id: any;
  model: Row;
};
export type PinnedRows = {
  top: RowEntry[];
  bottom: RowEntry[];
};
export declare const PinnedRows: {
  EMPTY: PinnedRows;
};
export type PinnedColumns = {
  left: Column[];
  right: Column[];
};
export declare const PinnedColumns: {
  EMPTY: PinnedColumns;
};
export type FocusedCell = {
  rowIndex: number;
  columnIndex: number;
  id?: any;
  field?: string;
};
export interface ColumnsRenderContext {
  firstColumnIndex: number;
  lastColumnIndex: number;
}
export interface RenderContext extends ColumnsRenderContext {
  firstRowIndex: number;
  lastRowIndex: number;
}
export interface GridScrollParams {
  left: number;
  top: number;
  renderContext?: RenderContext;
}
export type GridScrollFn = (v: GridScrollParams) => void;
export type PinnedRowPosition = keyof PinnedRows;
export type ScrollPosition = {
  top: number;
  left: number;
};
export declare const ScrollPosition: {
  EMPTY: {
    top: number;
    left: number;
  };
  equals: (a: ScrollPosition, b: ScrollPosition) => boolean;
};
export declare enum ScrollDirection {
  NONE = 0,
  UP = 1,
  DOWN = 2,
  LEFT = 3,
  RIGHT = 4,
}
export declare namespace ScrollDirection {
  function forDelta(dx: number, dy: number): ScrollDirection;
}
export type RowRange = {
  firstRowIndex: number;
  lastRowIndex: number;
};