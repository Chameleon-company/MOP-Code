import * as React from 'react';
import type { integer } from '@mui/x-internals/types';
import { Store } from '@mui/x-internals/store';
import type { CellColSpanInfo } from "../../models/colspan.js";
import { Dimensions } from "../dimensions.js";
import type { BaseState, ParamsWithDefaults } from "../../useVirtualizer.js";
import type { Layout } from "./layout.js";
import { RenderContext, ColumnsRenderContext, ColumnWithWidth, RowId, ScrollPosition } from "../../models/index.js";
export type VirtualizationParams = {
  /** @default false */
  isRtl?: boolean;
  /** The row buffer in pixels to render before and after the viewport.
   * @default 150 */
  rowBufferPx?: number;
  /** The column buffer in pixels to render before and after the viewport.
   * @default 150 */
  columnBufferPx?: number;
};
export type VirtualizationState<K extends string = string> = {
  enabled: boolean;
  enabledForRows: boolean;
  enabledForColumns: boolean;
  renderContext: RenderContext;
  props: Record<K, Record<string, any>>;
  context: Record<string, any>;
  scrollPosition: {
    current: ScrollPosition;
  };
};
export declare const EMPTY_RENDER_CONTEXT: {
  firstRowIndex: number;
  lastRowIndex: number;
  firstColumnIndex: number;
  lastColumnIndex: number;
};
export declare const Virtualization: {
  initialize: typeof initializeState;
  use: typeof useVirtualization;
  selectors: {
    store: (state: BaseState) => VirtualizationState<string>;
    renderContext: (state: BaseState) => RenderContext;
    enabledForRows: (state: BaseState) => boolean;
    enabledForColumns: (state: BaseState) => boolean;
    offsetTop: (args_0: Virtualization.State<Layout<{
      scroller: React.RefObject<HTMLElement | null>;
      container: React.RefObject<HTMLElement | null>;
    } & Record<string, React.RefObject<HTMLElement | null>>>> & Dimensions.State) => number;
    context: (state: BaseState) => Record<string, any>;
    scrollPosition: (state: BaseState) => {
      current: ScrollPosition;
    };
  };
};
export declare namespace Virtualization {
  type State<L extends Layout> = {
    virtualization: VirtualizationState<L extends Layout<infer E> ? keyof E : string>;
    getters: ReturnType<typeof useVirtualization>['getters'];
  };
  type API = ReturnType<typeof useVirtualization>;
}
declare function initializeState(params: ParamsWithDefaults): Virtualization.State<Layout<{
  scroller: React.RefObject<HTMLElement | null>;
  container: React.RefObject<HTMLElement | null>;
} & Record<string, React.RefObject<HTMLElement | null>>>>;
/** APIs to override for colspan/rowspan */
type AbstractAPI = {
  getCellColSpanInfo: (rowId: RowId, columnIndex: integer) => CellColSpanInfo;
  calculateColSpan: (rowId: RowId, minFirstColumn: integer, maxLastColumn: integer, columns: ColumnWithWidth[]) => void;
  getHiddenCellsOrigin: () => Record<RowId, Record<number, number>>;
};
type RequiredAPI = Dimensions.API & AbstractAPI;
export type VirtualizationLayoutParams = {
  containerRef: (node: HTMLDivElement | null) => void;
  scrollerRef: (node: HTMLDivElement | null) => void;
};
declare function useVirtualization(store: Store<BaseState>, params: ParamsWithDefaults, api: RequiredAPI): {
  getCellColSpanInfo: (rowId: RowId, columnIndex: integer) => CellColSpanInfo;
  calculateColSpan: (rowId: RowId, minFirstColumn: integer, maxLastColumn: integer, columns: ColumnWithWidth[]) => void;
  getHiddenCellsOrigin: () => Record<RowId, Record<number, number>>;
  getters: any;
  setPanels: React.Dispatch<React.SetStateAction<Readonly<Map<any, React.ReactNode>>>>;
  forceUpdateRenderContext: () => void;
  scheduleUpdateRenderContext: () => void;
};
export declare function areRenderContextsEqual(context1: RenderContext, context2: RenderContext): boolean;
export declare function computeOffsetLeft(columnPositions: number[], renderContext: ColumnsRenderContext, pinnedLeftLength: number): number;
export declare function roundToDecimalPlaces(value: number, decimals: number): number;
export {};