import { Store } from '@mui/x-internals/store';
import type { integer } from '@mui/x-internals/types';
import type { BaseState, ParamsWithDefaults } from "../useVirtualizer.js";
import type { ColumnWithWidth, RowId } from "../models/index.js";
import type { CellColSpanInfo } from "../models/colspan.js";
import { Virtualization } from "./virtualization/index.js";
type ColumnIndex = number;
type ColspanMap = Map<RowId, Record<ColumnIndex, CellColSpanInfo>>;
export type ColspanParams = {
  enabled: boolean;
  getColspan: (rowId: RowId, column: ColumnWithWidth, columnIndex: integer) => integer;
};
export declare const Colspan: {
  initialize: typeof initializeState;
  use: typeof useColspan;
  selectors: {};
};
export declare namespace Colspan {
  type State = {
    colspanMap: ColspanMap;
  };
  type API = ReturnType<typeof useColspan>;
}
declare function initializeState(_params: ParamsWithDefaults): {
  colspanMap: Map<any, any>;
};
declare function useColspan(store: Store<BaseState & Colspan.State>, params: ParamsWithDefaults, api: Virtualization.API): {
  resetColSpan: () => void;
  getCellColSpanInfo: (rowId: RowId, columnIndex: integer) => CellColSpanInfo | undefined;
  calculateColSpan: (rowId: RowId, minFirstColumn: integer, maxLastColumn: integer, columns: ColumnWithWidth[]) => void;
};
export {};