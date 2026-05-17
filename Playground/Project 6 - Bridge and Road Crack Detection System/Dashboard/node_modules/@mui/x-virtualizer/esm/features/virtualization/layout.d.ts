import * as React from 'react';
import { Store } from '@mui/x-internals/store';
import { Dimensions } from "../dimensions.js";
import { Virtualization, type VirtualizationLayoutParams } from "./virtualization.js";
import type { BaseState, ParamsWithDefaults } from "../../useVirtualizer.js";
type RequiredAPI = Dimensions.API;
type BaseElements = {
  scroller: React.RefObject<HTMLElement | null>;
  container: React.RefObject<HTMLElement | null>;
};
type AnyElements = BaseElements & Record<string, React.RefObject<HTMLElement | null>>;
export declare abstract class Layout<E extends AnyElements = AnyElements> {
  static elements: readonly (keyof AnyElements)[];
  refs: E;
  constructor(refs: E);
  abstract use(store: Store<BaseState>, params: ParamsWithDefaults, api: RequiredAPI, layoutParams: VirtualizationLayoutParams): any;
  refSetter(name: keyof E): (node: HTMLDivElement | null) => void;
}
type DataGridElements = BaseElements & {
  scrollbarVertical: React.RefObject<HTMLElement | null>;
  scrollbarHorizontal: React.RefObject<HTMLElement | null>;
};
export declare class LayoutDataGrid extends Layout<DataGridElements> {
  static elements: readonly ["scroller", "container", "content", "positioner", "scrollbarVertical", "scrollbarHorizontal"];
  use(store: Store<BaseState>, _params: ParamsWithDefaults, _api: RequiredAPI, layoutParams: VirtualizationLayoutParams): void;
  static selectors: {
    containerProps: (args_0: Virtualization.State<Layout<AnyElements>> & Dimensions.State) => {
      ref: any;
    };
    scrollerProps: (args_0: Virtualization.State<Layout<AnyElements>> & Dimensions.State) => {
      ref: any;
      style: {
        overflowX: string | undefined;
        overflowY: string | undefined;
      };
      role: string;
      tabIndex: number | undefined;
    };
    contentProps: (args_0: Virtualization.State<Layout<AnyElements>> & Dimensions.State) => {
      style: React.CSSProperties;
      role: string;
    };
    positionerProps: (args_0: Virtualization.State<Layout<AnyElements>> & Dimensions.State) => {
      style: {
        transform: string;
      };
    };
    scrollbarHorizontalProps: (args_0: Virtualization.State<Layout<AnyElements>> & Dimensions.State) => {
      ref: any;
      scrollPosition: {
        current: import("../../models/index.js").ScrollPosition;
      };
    };
    scrollbarVerticalProps: (args_0: Virtualization.State<Layout<AnyElements>> & Dimensions.State) => {
      ref: any;
      scrollPosition: {
        current: import("../../models/index.js").ScrollPosition;
      };
    };
    scrollAreaProps: (args_0: Virtualization.State<Layout<AnyElements>> & Dimensions.State) => {
      scrollPosition: {
        current: import("../../models/index.js").ScrollPosition;
      };
    };
  };
}
export declare class LayoutDataGridLegacy extends LayoutDataGrid {
  use(store: Store<BaseState>, _params: ParamsWithDefaults, _api: RequiredAPI, layoutParams: VirtualizationLayoutParams): {
    getContainerProps: () => {
      ref: any;
    };
    getScrollerProps: () => {
      ref: any;
      style: {
        overflowX: string | undefined;
        overflowY: string | undefined;
      };
      role: string;
      tabIndex: number | undefined;
    };
    getContentProps: () => {
      style: React.CSSProperties;
      role: string;
    };
    getPositionerProps: () => {
      style: {
        transform: string;
      };
    };
    getScrollbarVerticalProps: () => {
      ref: any;
      scrollPosition: {
        current: import("../../models/index.js").ScrollPosition;
      };
    };
    getScrollbarHorizontalProps: () => {
      ref: any;
      scrollPosition: {
        current: import("../../models/index.js").ScrollPosition;
      };
    };
    getScrollAreaProps: () => {
      scrollPosition: {
        current: import("../../models/index.js").ScrollPosition;
      };
    };
  };
}
type ListElements = BaseElements;
export declare class LayoutList extends Layout<ListElements> {
  static elements: readonly ["scroller", "container", "content", "positioner"];
  use(store: Store<BaseState>, _params: ParamsWithDefaults, _api: RequiredAPI, layoutParams: VirtualizationLayoutParams): void;
  static selectors: {
    containerProps: (args_0: Virtualization.State<Layout<AnyElements>> & Dimensions.State) => {
      ref: any;
      style: React.CSSProperties;
      role: string;
      tabIndex: number | undefined;
    };
    contentProps: (args_0: Virtualization.State<Layout<AnyElements>> & Dimensions.State) => {
      style: React.CSSProperties;
      role: string;
    };
    positionerProps: (args_0: Virtualization.State<Layout<AnyElements>> & Dimensions.State) => {
      style: React.CSSProperties;
    };
  };
}
export {};