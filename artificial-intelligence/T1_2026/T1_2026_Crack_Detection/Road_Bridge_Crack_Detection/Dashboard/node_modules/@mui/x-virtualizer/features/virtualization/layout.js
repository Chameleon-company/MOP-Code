"use strict";

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.LayoutList = exports.LayoutDataGridLegacy = exports.LayoutDataGrid = exports.Layout = void 0;
var _useForkRef = _interopRequireDefault(require("@mui/utils/useForkRef"));
var _useEventCallback = _interopRequireDefault(require("@mui/utils/useEventCallback"));
var platform = _interopRequireWildcard(require("@mui/x-internals/platform"));
var _store = require("@mui/x-internals/store");
var _dimensions = require("../../features/dimensions");
var _virtualization = require("./virtualization");
/* eslint-disable react-hooks/rules-of-hooks */

class Layout {
  static elements = ['scroller', 'container'];
  constructor(refs) {
    this.refs = refs;
  }
  refSetter(name) {
    return node => {
      if (node && this.refs[name].current !== node) {
        this.refs[name].current = node;
      }
    };
  }
}
exports.Layout = Layout;
class LayoutDataGrid extends Layout {
  static elements = ['scroller', 'container', 'content', 'positioner', 'scrollbarVertical', 'scrollbarHorizontal'];
  use(store, _params, _api, layoutParams) {
    const {
      scrollerRef,
      containerRef
    } = layoutParams;
    const scrollbarVerticalRef = (0, _useEventCallback.default)(this.refSetter('scrollbarVertical'));
    const scrollbarHorizontalRef = (0, _useEventCallback.default)(this.refSetter('scrollbarHorizontal'));
    store.state.virtualization.context = {
      scrollerRef,
      containerRef,
      scrollbarVerticalRef,
      scrollbarHorizontalRef
    };
  }
  static selectors = {
    containerProps: (0, _store.createSelectorMemoized)(_virtualization.Virtualization.selectors.context, context => ({
      ref: context.containerRef
    })),
    scrollerProps: (0, _store.createSelectorMemoized)(_virtualization.Virtualization.selectors.context, _dimensions.Dimensions.selectors.autoHeight, _dimensions.Dimensions.selectors.needsHorizontalScrollbar, (context, autoHeight, needsHorizontalScrollbar) => ({
      ref: context.scrollerRef,
      style: {
        overflowX: !needsHorizontalScrollbar ? 'hidden' : undefined,
        overflowY: autoHeight ? 'hidden' : undefined
      },
      role: 'presentation',
      // `tabIndex` shouldn't be used along role=presentation, but it fixes a Firefox bug
      // https://github.com/mui/mui-x/pull/13891#discussion_r1683416024
      tabIndex: platform.isFirefox ? -1 : undefined
    })),
    contentProps: (0, _store.createSelectorMemoized)(_dimensions.Dimensions.selectors.contentHeight, _dimensions.Dimensions.selectors.minimalContentHeight, _dimensions.Dimensions.selectors.columnsTotalWidth, _dimensions.Dimensions.selectors.needsHorizontalScrollbar, (contentHeight, minimalContentHeight, columnsTotalWidth, needsHorizontalScrollbar) => ({
      style: {
        width: needsHorizontalScrollbar ? columnsTotalWidth : 'auto',
        flexBasis: contentHeight === 0 ? minimalContentHeight : contentHeight,
        flexShrink: 0
      },
      role: 'presentation'
    })),
    positionerProps: (0, _store.createSelectorMemoized)(_virtualization.Virtualization.selectors.offsetTop, offsetTop => ({
      style: {
        transform: `translate3d(0, ${offsetTop}px, 0)`
      }
    })),
    scrollbarHorizontalProps: (0, _store.createSelectorMemoized)(_virtualization.Virtualization.selectors.context, _virtualization.Virtualization.selectors.scrollPosition, (context, scrollPosition) => ({
      ref: context.scrollbarHorizontalRef,
      scrollPosition
    })),
    scrollbarVerticalProps: (0, _store.createSelectorMemoized)(_virtualization.Virtualization.selectors.context, _virtualization.Virtualization.selectors.scrollPosition, (context, scrollPosition) => ({
      ref: context.scrollbarVerticalRef,
      scrollPosition
    })),
    scrollAreaProps: (0, _store.createSelectorMemoized)(_virtualization.Virtualization.selectors.scrollPosition, scrollPosition => ({
      scrollPosition
    }))
  };
}

// The current virtualizer API is exposed on one of the DataGrid slots, so we need to keep
// the old API for backward compatibility. This API prevents using fine-grained reactivity
// as all props are returned in a single object, so everything re-renders on any change.
//
// TODO(v9): Remove the legacy API.
exports.LayoutDataGrid = LayoutDataGrid;
class LayoutDataGridLegacy extends LayoutDataGrid {
  use(store, _params, _api, layoutParams) {
    super.use(store, _params, _api, layoutParams);
    const containerProps = store.use(LayoutDataGrid.selectors.containerProps);
    const scrollerProps = store.use(LayoutDataGrid.selectors.scrollerProps);
    const contentProps = store.use(LayoutDataGrid.selectors.contentProps);
    const positionerProps = store.use(LayoutDataGrid.selectors.positionerProps);
    const scrollbarVerticalProps = store.use(LayoutDataGrid.selectors.scrollbarVerticalProps);
    const scrollbarHorizontalProps = store.use(LayoutDataGrid.selectors.scrollbarHorizontalProps);
    const scrollAreaProps = store.use(LayoutDataGrid.selectors.scrollAreaProps);
    return {
      getContainerProps: () => containerProps,
      getScrollerProps: () => scrollerProps,
      getContentProps: () => contentProps,
      getPositionerProps: () => positionerProps,
      getScrollbarVerticalProps: () => scrollbarVerticalProps,
      getScrollbarHorizontalProps: () => scrollbarHorizontalProps,
      getScrollAreaProps: () => scrollAreaProps
    };
  }
}
exports.LayoutDataGridLegacy = LayoutDataGridLegacy;
class LayoutList extends Layout {
  static elements = ['scroller', 'container', 'content', 'positioner'];
  use(store, _params, _api, layoutParams) {
    const {
      scrollerRef,
      containerRef
    } = layoutParams;
    const mergedRef = (0, _useForkRef.default)(scrollerRef, containerRef);
    store.state.virtualization.context = {
      mergedRef
    };
  }
  static selectors = {
    containerProps: (0, _store.createSelectorMemoized)(_virtualization.Virtualization.selectors.context, _dimensions.Dimensions.selectors.autoHeight, _dimensions.Dimensions.selectors.needsHorizontalScrollbar, (context, autoHeight, needsHorizontalScrollbar) => ({
      ref: context.mergedRef,
      style: {
        overflowX: !needsHorizontalScrollbar ? 'hidden' : undefined,
        overflowY: autoHeight ? 'hidden' : undefined,
        position: 'relative'
      },
      role: 'presentation',
      // `tabIndex` shouldn't be used along role=presentation, but it fixes a Firefox bug
      // https://github.com/mui/mui-x/pull/13891#discussion_r1683416024
      tabIndex: platform.isFirefox ? -1 : undefined
    })),
    contentProps: (0, _store.createSelectorMemoized)(_dimensions.Dimensions.selectors.contentHeight, contentHeight => {
      return {
        style: {
          position: 'absolute',
          display: 'inline-block',
          width: '100%',
          height: contentHeight,
          top: 0,
          left: 0,
          zIndex: -1
        },
        role: 'presentation'
      };
    }),
    positionerProps: (0, _store.createSelectorMemoized)(_virtualization.Virtualization.selectors.offsetTop, offsetTop => ({
      style: {
        height: offsetTop
      }
    }))
  };
}
exports.LayoutList = LayoutList;