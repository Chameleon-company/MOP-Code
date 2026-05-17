import { Store } from '@mui/x-internals/store';
import type { BaseState, ParamsWithDefaults } from "../useVirtualizer.js";
import type { RowSpanningState } from "../models/rowspan.js";
import { Virtualization } from "./virtualization/index.js";
export declare const Rowspan: {
  initialize: typeof initializeState;
  use: typeof useRowspan;
  selectors: {
    state: (state: Rowspan.State) => RowSpanningState;
    hiddenCells: (state: Rowspan.State) => Record<any, Record<number, boolean>>;
    spannedCells: (state: Rowspan.State) => Record<any, Record<number, number>>;
    hiddenCellsOriginMap: (state: Rowspan.State) => Record<number, Record<number, number>>;
  };
};
export declare namespace Rowspan {
  type State = {
    rowSpanning: RowSpanningState;
  };
  type API = ReturnType<typeof useRowspan>;
}
declare function initializeState(params: ParamsWithDefaults): Rowspan.State;
declare function useRowspan(store: Store<BaseState & Rowspan.State>, _params: ParamsWithDefaults, _api: Virtualization.API): {
  getHiddenCellsOrigin: () => Record<number, Record<number, number>>;
};
export {};