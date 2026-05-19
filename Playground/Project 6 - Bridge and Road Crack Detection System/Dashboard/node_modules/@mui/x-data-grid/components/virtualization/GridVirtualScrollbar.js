"use strict";
'use client';

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.scrollbarSizeCssExpression = exports.ScrollbarCorner = exports.GridVirtualScrollbar = void 0;
var React = _interopRequireWildcard(require("react"));
var _styles = require("@mui/material/styles");
var _useEventCallback = _interopRequireDefault(require("@mui/utils/useEventCallback"));
var _useForkRef = _interopRequireDefault(require("@mui/utils/useForkRef"));
var _composeClasses = _interopRequireDefault(require("@mui/utils/composeClasses"));
var _forwardRef = require("@mui/x-internals/forwardRef");
var _useOnMount = require("../../hooks/utils/useOnMount");
var _useGridPrivateApiContext = require("../../hooks/utils/useGridPrivateApiContext");
var _hooks = require("../../hooks");
var _useGridRootProps = require("../../hooks/utils/useGridRootProps");
var _gridClasses = require("../../constants/gridClasses");
var _jsxRuntime = require("react/jsx-runtime");
const useUtilityClasses = (ownerState, position) => {
  const {
    classes
  } = ownerState;
  const slots = {
    root: ['scrollbar', `scrollbar--${position}`],
    content: ['scrollbarContent']
  };
  return (0, _composeClasses.default)(slots, _gridClasses.getDataGridUtilityClass, classes);
};

// In macOS Safari and Gnome Web, scrollbars are overlaid and don't affect the layout. So we consider
// their size to be 0px throughout all the calculations, but the floating scrollbar container does need
// to appear and have a real size. We set it to 14px because it seems like an acceptable value and we
// don't have a method to find the required size for scrollbars on those platforms.
const scrollbarSizeCssExpression = exports.scrollbarSizeCssExpression = 'calc(max(var(--DataGrid-scrollbarSize), 14px))';
const Scrollbar = (0, _styles.styled)('div', {
  slot: 'internal',
  shouldForwardProp: undefined
})({
  position: 'absolute',
  display: 'inline-block',
  zIndex: 60,
  '&:hover': {
    zIndex: 70
  },
  '--size': scrollbarSizeCssExpression
});
const ScrollbarVertical = (0, _styles.styled)(Scrollbar, {
  slot: 'internal'
})({
  width: 'var(--size)',
  height: 'calc(var(--DataGrid-hasScrollY) * (100% - var(--DataGrid-headersTotalHeight) - var(--DataGrid-hasScrollX) * var(--DataGrid-scrollbarSize)))',
  overflowY: 'auto',
  overflowX: 'hidden',
  // Disable focus-visible style, it's a scrollbar.
  outline: 0,
  '& > div': {
    width: 'var(--size)'
  },
  top: 'var(--DataGrid-headersTotalHeight)',
  right: 0
});
const ScrollbarHorizontal = (0, _styles.styled)(Scrollbar, {
  slot: 'internal'
})({
  width: 'calc(var(--DataGrid-hasScrollX) * (100% - var(--DataGrid-hasScrollY) * var(--DataGrid-scrollbarSize)))',
  height: 'var(--size)',
  overflowY: 'hidden',
  overflowX: 'auto',
  // Disable focus-visible style, it's a scrollbar.
  outline: 0,
  '& > div': {
    height: 'var(--size)'
  },
  bottom: 0
});
const ScrollbarCorner = exports.ScrollbarCorner = (0, _styles.styled)(Scrollbar, {
  slot: 'internal'
})({
  width: 'var(--size)',
  height: 'var(--size)',
  right: 0,
  bottom: 0,
  overflow: 'scroll',
  '@media print': {
    display: 'none'
  }
});
const GridVirtualScrollbar = exports.GridVirtualScrollbar = (0, _forwardRef.forwardRef)(function GridVirtualScrollbar(props, ref) {
  const apiRef = (0, _useGridPrivateApiContext.useGridPrivateApiContext)();
  const rootProps = (0, _useGridRootProps.useGridRootProps)();
  const isLocked = React.useRef(false);
  const lastPosition = React.useRef(0);
  const scrollbarRef = React.useRef(null);
  const classes = useUtilityClasses(rootProps, props.position);
  const dimensions = (0, _hooks.useGridSelector)(apiRef, _hooks.gridDimensionsSelector);
  const propertyDimension = props.position === 'vertical' ? 'height' : 'width';
  const propertyScroll = props.position === 'vertical' ? 'scrollTop' : 'scrollLeft';
  const propertyScrollPosition = props.position === 'vertical' ? 'top' : 'left';
  const scrollbarInnerSize = props.position === 'horizontal' ? dimensions.minimumSize.width : dimensions.minimumSize.height - dimensions.headersTotalHeight;
  const onScrollerScroll = (0, _useEventCallback.default)(() => {
    const scrollbar = scrollbarRef.current;
    const scrollPosition = props.scrollPosition.current;
    if (!scrollbar) {
      return;
    }
    if (scrollPosition[propertyScrollPosition] === lastPosition.current) {
      return;
    }
    lastPosition.current = scrollPosition[propertyScrollPosition];
    if (isLocked.current) {
      isLocked.current = false;
      return;
    }
    isLocked.current = true;
    scrollbar[propertyScroll] = props.scrollPosition.current[propertyScrollPosition];
  });
  const onScrollbarScroll = (0, _useEventCallback.default)(() => {
    const scroller = apiRef.current.virtualScrollerRef.current;
    const scrollbar = scrollbarRef.current;
    if (!scrollbar) {
      return;
    }
    if (isLocked.current) {
      isLocked.current = false;
      return;
    }
    isLocked.current = true;
    scroller[propertyScroll] = scrollbar[propertyScroll];
  });
  (0, _useOnMount.useOnMount)(() => {
    const scroller = apiRef.current.virtualScrollerRef.current;
    const scrollbar = scrollbarRef.current;
    const options = {
      passive: true
    };
    scroller.addEventListener('scroll', onScrollerScroll, options);
    scrollbar.addEventListener('scroll', onScrollbarScroll, options);
    return () => {
      scroller.removeEventListener('scroll', onScrollerScroll, options);
      scrollbar.removeEventListener('scroll', onScrollbarScroll, options);
    };
  });
  const Container = props.position === 'vertical' ? ScrollbarVertical : ScrollbarHorizontal;
  const scrollbarInnerStyle = React.useMemo(() => ({
    [propertyDimension]: `${scrollbarInnerSize}px`
  }), [propertyDimension, scrollbarInnerSize]);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(Container, {
    ref: (0, _useForkRef.default)(ref, scrollbarRef),
    className: classes.root,
    tabIndex: -1,
    "aria-hidden": "true"
    // tabIndex does not prevent focus with a mouse click, throwing a console error
    // https://github.com/mui/mui-x/issues/16706
    ,
    onFocus: event => {
      event.target.blur();
    },
    children: /*#__PURE__*/(0, _jsxRuntime.jsx)("div", {
      className: classes.content,
      style: scrollbarInnerStyle
    })
  });
});
if (process.env.NODE_ENV !== "production") GridVirtualScrollbar.displayName = "GridVirtualScrollbar";