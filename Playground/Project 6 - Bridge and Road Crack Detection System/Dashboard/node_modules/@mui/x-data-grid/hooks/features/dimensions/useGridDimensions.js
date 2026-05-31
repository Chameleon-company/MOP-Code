"use strict";
'use client';

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.dimensionsStateInitializer = void 0;
exports.useGridDimensions = useGridDimensions;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var React = _interopRequireWildcard(require("react"));
var _store = require("@mui/x-internals/store");
var _useGridEvent = require("../../utils/useGridEvent");
var _useGridApiMethod = require("../../utils/useGridApiMethod");
var _createSelector = require("../../../utils/createSelector");
var _useGridLogger = require("../../utils/useGridLogger");
var _columns = require("../columns");
var _gridDimensionsSelectors = require("./gridDimensionsSelectors");
var _density = require("../density");
var _gridRowsUtils = require("../rows/gridRowsUtils");
var _gridColumnsUtils = require("../columns/gridColumnsUtils");
var _dataGridPropsDefaultValues = require("../../../constants/dataGridPropsDefaultValues");
var _roundToDecimalPlaces = require("../../../utils/roundToDecimalPlaces");
var _isJSDOM = require("../../../utils/isJSDOM");
const EMPTY_SIZE = {
  width: 0,
  height: 0
};
const EMPTY_DIMENSIONS = {
  isReady: false,
  root: EMPTY_SIZE,
  viewportOuterSize: EMPTY_SIZE,
  viewportInnerSize: EMPTY_SIZE,
  contentSize: EMPTY_SIZE,
  minimumSize: EMPTY_SIZE,
  hasScrollX: false,
  hasScrollY: false,
  scrollbarSize: 0,
  headerHeight: 0,
  groupHeaderHeight: 0,
  headerFilterHeight: 0,
  rowWidth: 0,
  rowHeight: 0,
  columnsTotalWidth: 0,
  leftPinnedWidth: 0,
  rightPinnedWidth: 0,
  headersTotalHeight: 0,
  topContainerHeight: 0,
  bottomContainerHeight: 0,
  autoHeight: false,
  minimalContentHeight: undefined
};
const dimensionsStateInitializer = (state, props, apiRef) => {
  const dimensions = EMPTY_DIMENSIONS;
  const density = (0, _density.gridDensityFactorSelector)(apiRef);
  const dimensionsWithStatic = (0, _extends2.default)({}, dimensions, getStaticDimensions(props, apiRef, density, (0, _columns.gridVisiblePinnedColumnDefinitionsSelector)(apiRef)));
  apiRef.current.store.state.dimensions = dimensionsWithStatic;
  return (0, _extends2.default)({}, state, {
    dimensions: dimensionsWithStatic
  });
};
exports.dimensionsStateInitializer = dimensionsStateInitializer;
const columnsTotalWidthSelector = (0, _createSelector.createSelector)(_columns.gridVisibleColumnDefinitionsSelector, _columns.gridColumnPositionsSelector, (visibleColumns, positions) => {
  const colCount = visibleColumns.length;
  if (colCount === 0) {
    return 0;
  }
  return (0, _roundToDecimalPlaces.roundToDecimalPlaces)(positions[colCount - 1] + visibleColumns[colCount - 1].computedWidth, 1);
});
function useGridDimensions(apiRef, props) {
  const getRootDimensions = React.useCallback(() => (0, _gridDimensionsSelectors.gridDimensionsSelector)(apiRef), [apiRef]);
  const apiPublic = {
    getRootDimensions
  };
  const apiPrivate = {
    updateDimensions: () => {
      return apiRef.current.virtualizer.api.updateDimensions();
    },
    getViewportPageSize: () => {
      return apiRef.current.virtualizer.api.getViewportPageSize();
    }
  };
  (0, _useGridApiMethod.useGridApiMethod)(apiRef, apiPublic, 'public');
  (0, _useGridApiMethod.useGridApiMethod)(apiRef, apiPrivate, 'private');
  const handleRootMount = root => {
    setCSSVariables(root, (0, _gridDimensionsSelectors.gridDimensionsSelector)(apiRef));
  };
  (0, _useGridEvent.useGridEventPriority)(apiRef, 'rootMount', handleRootMount);
  (0, _useGridEvent.useGridEventPriority)(apiRef, 'debouncedResize', props.onResize);
  if (process.env.NODE_ENV !== 'production') {
    /* eslint-disable react-hooks/rules-of-hooks */
    const logger = (0, _useGridLogger.useGridLogger)(apiRef, 'useResizeContainer');
    const errorShown = React.useRef(false);
    (0, _useGridEvent.useGridEventPriority)(apiRef, 'resize', size => {
      if (!getRootDimensions().isReady) {
        return;
      }
      if (size.height === 0 && !errorShown.current && !props.autoHeight && !_isJSDOM.isJSDOM) {
        logger.error(['The parent DOM element of the Data Grid has an empty height.', 'Please make sure that this element has an intrinsic height.', 'The grid displays with a height of 0px.', '', 'More details: https://mui.com/r/x-data-grid-no-dimensions.'].join('\n'));
        errorShown.current = true;
      }
      if (size.width === 0 && !errorShown.current && !_isJSDOM.isJSDOM) {
        logger.error(['The parent DOM element of the Data Grid has an empty width.', 'Please make sure that this element has an intrinsic width.', 'The grid displays with a width of 0px.', '', 'More details: https://mui.com/r/x-data-grid-no-dimensions.'].join('\n'));
        errorShown.current = true;
      }
    });
    /* eslint-enable react-hooks/rules-of-hooks */
  }
  (0, _store.useStoreEffect)(apiRef.current.store, s => s.dimensions, (previous, next) => {
    if (!next.isReady) {
      return;
    }
    if (apiRef.current.rootElementRef.current) {
      setCSSVariables(apiRef.current.rootElementRef.current, next);
    }
    if (!areElementSizesEqual(next.viewportInnerSize, previous.viewportInnerSize)) {
      apiRef.current.publishEvent('viewportInnerSizeChange', next.viewportInnerSize);
    }
    apiRef.current.publishEvent('debouncedResize', next.root);
  });
}
function setCSSVariables(root, dimensions) {
  const set = (k, v) => root.style.setProperty(k, v);
  set('--DataGrid-hasScrollX', `${Number(dimensions.hasScrollX)}`);
  set('--DataGrid-hasScrollY', `${Number(dimensions.hasScrollY)}`);
  set('--DataGrid-scrollbarSize', `${dimensions.scrollbarSize}px`);
  set('--DataGrid-rowWidth', `${dimensions.rowWidth}px`);
  set('--DataGrid-columnsTotalWidth', `${dimensions.columnsTotalWidth}px`);
  set('--DataGrid-leftPinnedWidth', `${dimensions.leftPinnedWidth}px`);
  set('--DataGrid-rightPinnedWidth', `${dimensions.rightPinnedWidth}px`);
  set('--DataGrid-headerHeight', `${dimensions.headerHeight}px`);
  set('--DataGrid-headersTotalHeight', `${dimensions.headersTotalHeight}px`);
  set('--DataGrid-topContainerHeight', `${dimensions.topContainerHeight}px`);
  set('--DataGrid-bottomContainerHeight', `${dimensions.bottomContainerHeight}px`);
  set('--height', `${dimensions.rowHeight}px`);
}
function getStaticDimensions(props, apiRef, density, pinnedColumnns) {
  const validRowHeight = (0, _gridRowsUtils.getValidRowHeight)(props.rowHeight, _dataGridPropsDefaultValues.DATA_GRID_PROPS_DEFAULT_VALUES.rowHeight, _gridRowsUtils.rowHeightWarning);
  return {
    rowHeight: Math.floor(validRowHeight * density),
    headerHeight: Math.floor(props.columnHeaderHeight * density),
    groupHeaderHeight: Math.floor((props.columnGroupHeaderHeight ?? props.columnHeaderHeight) * density),
    headerFilterHeight: Math.floor((props.headerFilterHeight ?? props.columnHeaderHeight) * density),
    columnsTotalWidth: columnsTotalWidthSelector(apiRef),
    headersTotalHeight: (0, _gridColumnsUtils.getTotalHeaderHeight)(apiRef, props),
    leftPinnedWidth: pinnedColumnns.left.reduce((w, col) => w + col.computedWidth, 0),
    rightPinnedWidth: pinnedColumnns.right.reduce((w, col) => w + col.computedWidth, 0)
  };
}
function areElementSizesEqual(a, b) {
  return a.width === b.width && a.height === b.height;
}