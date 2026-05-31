"use strict";
'use client';

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.GridScrollShadows = GridScrollShadows;
var React = _interopRequireWildcard(require("react"));
var _styles = require("@mui/material/styles");
var _RtlProvider = require("@mui/system/RtlProvider");
var _composeClasses = _interopRequireDefault(require("@mui/utils/composeClasses"));
var _hooks = require("../hooks");
var _gridRowsSelector = require("../hooks/features/rows/gridRowsSelector");
var _useGridRootProps = require("../hooks/utils/useGridRootProps");
var _cssVariables = require("../constants/cssVariables");
var _useGridPrivateApiContext = require("../hooks/utils/useGridPrivateApiContext");
var _gridDimensionsSelectors = require("../hooks/features/dimensions/gridDimensionsSelectors");
var _constants = require("../constants");
var _jsxRuntime = require("react/jsx-runtime");
const useUtilityClasses = ownerState => {
  const {
    classes,
    position
  } = ownerState;
  const slots = {
    root: ['scrollShadow', `scrollShadow--${position}`]
  };
  return (0, _composeClasses.default)(slots, _constants.getDataGridUtilityClass, classes);
};
const ScrollShadow = (0, _styles.styled)('div', {
  name: 'MuiDataGrid',
  slot: 'ScrollShadow',
  overridesResolver: (props, styles) => [styles.root, styles[props.position]]
})(({
  theme
}) => ({
  position: 'absolute',
  inset: 0,
  pointerEvents: 'none',
  transition: _cssVariables.vars.transition(['box-shadow'], {
    duration: _cssVariables.vars.transitions.duration.short
  }),
  '--length': theme.palette.mode === 'dark' ? '8px' : '6px',
  '--length-inverse': 'calc(var(--length) * -1)',
  '--opacity': theme.palette.mode === 'dark' ? '0.7' : '0.18',
  '--blur': 'var(--length)',
  '--spread': 'calc(var(--length) * -1)',
  '--color': '0, 0, 0',
  '--color-start': 'rgba(var(--color), calc(var(--hasScrollStart) * var(--opacity)))',
  '--color-end': 'rgba(var(--color), calc(var(--hasScrollEnd) * var(--opacity)))',
  variants: [{
    props: {
      position: 'vertical'
    },
    style: {
      top: 'var(--DataGrid-topContainerHeight)',
      bottom: 'calc(var(--DataGrid-bottomContainerHeight) + var(--DataGrid-hasScrollX) * var(--DataGrid-scrollbarSize))',
      boxShadow: 'inset 0 var(--length) var(--blur) var(--spread) var(--color-start), inset 0 var(--length-inverse) var(--blur) var(--spread) var(--color-end)'
    }
  }, {
    props: {
      position: 'horizontal'
    },
    style: {
      left: 'var(--DataGrid-leftPinnedWidth)',
      right: 'calc(var(--DataGrid-rightPinnedWidth) + var(--DataGrid-hasScrollY) * var(--DataGrid-scrollbarSize))',
      boxShadow: 'inset var(--length) 0 var(--blur) var(--spread) var(--color-start), inset var(--length-inverse) 0 var(--blur) var(--spread) var(--color-end)'
    }
  }]
}));
function GridScrollShadows(props) {
  const {
    position
  } = props;
  const rootProps = (0, _useGridRootProps.useGridRootProps)();
  const ownerState = {
    classes: rootProps.classes,
    position
  };
  const classes = useUtilityClasses(ownerState);
  const ref = React.useRef(null);
  const apiRef = (0, _useGridPrivateApiContext.useGridPrivateApiContext)();
  const hasScrollX = (0, _hooks.useGridSelector)(apiRef, _gridDimensionsSelectors.gridHasScrollXSelector);
  const hasScrollY = (0, _hooks.useGridSelector)(apiRef, _gridDimensionsSelectors.gridHasScrollYSelector);
  const pinnedRows = (0, _hooks.useGridSelector)(apiRef, _gridRowsSelector.gridPinnedRowsSelector);
  const pinnedColumns = (0, _hooks.useGridSelector)(apiRef, _hooks.gridPinnedColumnsSelector);
  const initialScrollable = position === 'vertical' ? hasScrollY && pinnedRows?.bottom?.length > 0 : hasScrollX && pinnedColumns?.right?.length !== undefined && pinnedColumns?.right?.length > 0;
  const isRtl = (0, _RtlProvider.useRtl)();
  const updateScrollShadowVisibility = React.useCallback(scrollPosition => {
    if (!ref.current) {
      return;
    }
    // Math.abs to convert negative scroll position (RTL) to positive
    const scroll = Math.abs(Math.round(scrollPosition));
    const dimensions = (0, _hooks.gridDimensionsSelector)(apiRef);
    const maxScroll = Math.round(dimensions.contentSize[position === 'vertical' ? 'height' : 'width'] - dimensions.viewportInnerSize[position === 'vertical' ? 'height' : 'width']);
    const hasPinnedStart = position === 'vertical' ? pinnedRows?.top?.length > 0 : pinnedColumns?.left?.length !== undefined && pinnedColumns?.left?.length > 0;
    const hasPinnedEnd = position === 'vertical' ? pinnedRows?.bottom?.length > 0 : pinnedColumns?.right?.length !== undefined && pinnedColumns?.right?.length > 0;
    const scrollIsNotAtStart = isRtl ? scroll < maxScroll : scroll > 0;
    const scrollIsNotAtEnd = isRtl ? scroll > 0 : scroll < maxScroll;
    ref.current.style.setProperty('--hasScrollStart', hasPinnedStart && scrollIsNotAtStart ? '1' : '0');
    ref.current.style.setProperty('--hasScrollEnd', hasPinnedEnd && scrollIsNotAtEnd ? '1' : '0');
  }, [pinnedRows, pinnedColumns, isRtl, position, apiRef]);
  const handleScrolling = scrollParams => {
    updateScrollShadowVisibility(scrollParams[position === 'vertical' ? 'top' : 'left']);
  };
  const handleColumnResizeStop = () => {
    if (position === 'horizontal') {
      updateScrollShadowVisibility(apiRef.current.virtualScrollerRef?.current?.scrollLeft || 0);
    }
  };
  (0, _hooks.useGridEvent)(apiRef, 'scrollPositionChange', handleScrolling);
  (0, _hooks.useGridEvent)(apiRef, 'columnResizeStop', handleColumnResizeStop);
  React.useEffect(() => {
    updateScrollShadowVisibility((position === 'horizontal' ? apiRef.current.virtualScrollerRef?.current?.scrollLeft : apiRef.current.virtualScrollerRef?.current?.scrollTop) ?? 0);
  }, [updateScrollShadowVisibility, apiRef, position]);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(ScrollShadow, {
    className: classes.root,
    ownerState: ownerState,
    ref: ref,
    style: {
      '--hasScrollStart': 0,
      '--hasScrollEnd': initialScrollable ? '1' : '0'
    }
  });
}