"use strict";

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useGridVirtualizer = useGridVirtualizer;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var React = _interopRequireWildcard(require("react"));
var _useLazyRef = _interopRequireDefault(require("@mui/utils/useLazyRef"));
var _useEventCallback = _interopRequireDefault(require("@mui/utils/useEventCallback"));
var _RtlProvider = require("@mui/system/RtlProvider");
var _math = require("@mui/x-internals/math");
var _lruMemoize = require("@mui/x-internals/lruMemoize");
var _store = require("@mui/x-internals/store");
var _xVirtualizer = require("@mui/x-virtualizer");
var _useFirstRender = require("../utils/useFirstRender");
var _createSelector = require("../../utils/createSelector");
var _useGridSelector = require("../utils/useGridSelector");
var _gridDimensionsSelectors = require("../features/dimensions/gridDimensionsSelectors");
var _density = require("../features/density");
var _gridColumnsSelector = require("../features/columns/gridColumnsSelector");
var _gridRowsSelector = require("../features/rows/gridRowsSelector");
var _useGridVisibleRows = require("../utils/useGridVisibleRows");
var _pagination = require("../features/pagination");
var _gridFocusedVirtualCellSelector = require("../features/virtualization/gridFocusedVirtualCellSelector");
var _rowSelection = require("../features/rowSelection");
var _dataGridPropsDefaultValues = require("../../constants/dataGridPropsDefaultValues");
var _gridRowsUtils = require("../features/rows/gridRowsUtils");
var _gridColumnsUtils = require("../features/columns/gridColumnsUtils");
var _useGridOverlays = require("../features/overlays/useGridOverlays");
var _useGridRootProps = require("../utils/useGridRootProps");
var _useGridPrivateApiContext = require("../utils/useGridPrivateApiContext");
var _useGridRowsMeta = require("../features/rows/useGridRowsMeta");
var _utils = require("../../utils/utils");
var _jsxRuntime = require("react/jsx-runtime");
const columnsTotalWidthSelector = (0, _createSelector.createSelector)(_gridColumnsSelector.gridVisibleColumnDefinitionsSelector, _gridColumnsSelector.gridColumnPositionsSelector, (visibleColumns, positions) => {
  const colCount = visibleColumns.length;
  if (colCount === 0) {
    return 0;
  }
  return (0, _math.roundToDecimalPlaces)(positions[colCount - 1] + visibleColumns[colCount - 1].computedWidth, 1);
});

/** Translates virtualizer state to grid state */
const addGridDimensionsCreator = () => (0, _lruMemoize.lruMemoize)((dimensions, headerHeight, groupHeaderHeight, headerFilterHeight, headersTotalHeight) => {
  return (0, _extends2.default)({}, dimensions, {
    headerHeight,
    groupHeaderHeight,
    headerFilterHeight,
    headersTotalHeight
  });
}, {
  maxSize: 1
});

/**
 * Virtualizer setup
 */
function useGridVirtualizer() {
  const isRtl = (0, _RtlProvider.useRtl)();
  const rootProps = (0, _useGridRootProps.useGridRootProps)();
  const apiRef = (0, _useGridPrivateApiContext.useGridPrivateApiContext)();
  const {
    listView
  } = rootProps;
  const visibleColumns = (0, _useGridSelector.useGridSelector)(apiRef, _gridColumnsSelector.gridVisibleColumnDefinitionsSelector);
  const pinnedRows = (0, _useGridSelector.useGridSelector)(apiRef, _gridRowsSelector.gridPinnedRowsSelector);
  const pinnedColumns = (0, _gridColumnsSelector.gridVisiblePinnedColumnDefinitionsSelector)(apiRef);
  const rowSelectionManager = (0, _useGridSelector.useGridSelector)(apiRef, _rowSelection.gridRowSelectionManagerSelector);
  const isRowSelected = React.useCallback(id => rowSelectionManager.has(id) && apiRef.current.isRowSelectable(id), [rowSelectionManager, apiRef]);
  const currentPage = (0, _useGridVisibleRows.useGridVisibleRows)(apiRef);
  const hasColSpan = (0, _useGridSelector.useGridSelector)(apiRef, _gridColumnsSelector.gridHasColSpanSelector);
  const verticalScrollbarWidth = (0, _useGridSelector.useGridSelector)(apiRef, _gridDimensionsSelectors.gridVerticalScrollbarWidthSelector);
  const hasFiller = (0, _useGridSelector.useGridSelector)(apiRef, _gridDimensionsSelectors.gridHasFillerSelector);
  const {
    autoHeight
  } = rootProps;
  const scrollReset = listView;

  // <DIMENSIONS>
  const density = (0, _useGridSelector.useGridSelector)(apiRef, _density.gridDensityFactorSelector);
  const baseRowHeight = (0, _gridRowsUtils.getValidRowHeight)(rootProps.rowHeight, _dataGridPropsDefaultValues.DATA_GRID_PROPS_DEFAULT_VALUES.rowHeight, _gridRowsUtils.rowHeightWarning);
  const rowHeight = Math.floor(baseRowHeight * density);
  const headerHeight = Math.floor(rootProps.columnHeaderHeight * density);
  const groupHeaderHeight = Math.floor((rootProps.columnGroupHeaderHeight ?? rootProps.columnHeaderHeight) * density);
  const headerFilterHeight = Math.floor((rootProps.headerFilterHeight ?? rootProps.columnHeaderHeight) * density);
  const columnsTotalWidth = (0, _useGridSelector.useGridSelector)(apiRef, columnsTotalWidthSelector);
  const headersTotalHeight = (0, _gridColumnsUtils.getTotalHeaderHeight)(apiRef, rootProps);
  const leftPinnedWidth = pinnedColumns.left.reduce((w, col) => w + col.computedWidth, 0);
  const rightPinnedWidth = pinnedColumns.right.reduce((w, col) => w + col.computedWidth, 0);
  const overlayState = (0, _useGridOverlays.useGridOverlays)(apiRef, rootProps);
  const dimensionsParams = {
    rowHeight,
    headerHeight,
    columnsTotalWidth,
    leftPinnedWidth,
    rightPinnedWidth,
    topPinnedHeight: headersTotalHeight,
    bottomPinnedHeight: 0,
    autoHeight,
    minimalContentHeight: _gridRowsUtils.minimalContentHeight,
    scrollbarSize: rootProps.scrollbarSize
  };
  const addGridDimensions = (0, _useLazyRef.default)(addGridDimensionsCreator).current;

  // </DIMENSIONS>

  // <ROWS_META>
  const dataRowCount = (0, _useGridSelector.useGridSelector)(apiRef, _gridRowsSelector.gridRowCountSelector);
  const pagination = (0, _useGridSelector.useGridSelector)(apiRef, _pagination.gridPaginationSelector);
  const rowCount = Math.min(pagination.enabled ? pagination.paginationModel.pageSize : dataRowCount, dataRowCount);
  const {
    getRowHeight,
    getEstimatedRowHeight,
    getRowSpacing
  } = rootProps;
  // </ROWS_META>

  const RowSlot = rootProps.slots.row;
  const rowSlotProps = rootProps.slotProps?.row;
  const focusedVirtualCell = (0, _useGridSelector.useGridSelector)(apiRef, _gridFocusedVirtualCellSelector.gridFocusedVirtualCellSelector);
  // We need it to trigger a new render, but rowsMeta needs access to the latest value, hence we cannot pass it to the focusedVirtualCell callback in the virtualizer params
  (0, _utils.eslintUseValue)(focusedVirtualCell);
  const layout = (0, _useLazyRef.default)(() => new _xVirtualizer.LayoutDataGridLegacy({
    container: apiRef.current.mainElementRef,
    scroller: apiRef.current.virtualScrollerRef,
    scrollbarVertical: apiRef.current.virtualScrollbarVerticalRef,
    scrollbarHorizontal: apiRef.current.virtualScrollbarHorizontalRef
  })).current;
  const virtualizer = (0, _xVirtualizer.useVirtualizer)({
    layout,
    dimensions: dimensionsParams,
    virtualization: {
      isRtl,
      rowBufferPx: rootProps.rowBufferPx,
      columnBufferPx: rootProps.columnBufferPx
    },
    colspan: {
      enabled: hasColSpan,
      getColspan: React.useCallback((rowId, column) => {
        if (typeof column.colSpan === 'function') {
          const row = apiRef.current.getRow(rowId);
          const value = apiRef.current.getRowValue(row, column);
          return column.colSpan(value, row, column, apiRef) ?? 0;
        }
        return column.colSpan ?? 1;
      }, [apiRef])
    },
    initialState: {
      scroll: rootProps.initialState?.scroll,
      rowSpanning: apiRef.current.state.rowSpanning,
      virtualization: apiRef.current.state.virtualization
    },
    rows: currentPage.rows,
    range: currentPage.range,
    rowCount,
    columns: visibleColumns,
    pinnedRows,
    pinnedColumns,
    disableHorizontalScroll: listView,
    disableVerticalScroll: overlayState.overlayType === 'noColumnsOverlay' || overlayState.loadingOverlayVariant === 'skeleton',
    getRowHeight: React.useMemo(() => {
      if (!getRowHeight) {
        return undefined;
      }
      return rowEntry => getRowHeight((0, _extends2.default)({}, rowEntry, {
        densityFactor: density
      }));
    }, [getRowHeight, density]),
    getEstimatedRowHeight: React.useMemo(() => getEstimatedRowHeight ? rowEntry => getEstimatedRowHeight((0, _extends2.default)({}, rowEntry, {
      densityFactor: density
    })) : undefined, [getEstimatedRowHeight, density]),
    getRowSpacing: React.useMemo(() => getRowSpacing ? rowEntry => {
      const indexRelativeToCurrentPage = currentPage.rowIdToIndexMap.get(rowEntry.id) ?? -1;
      const visibility = {
        isFirstVisible: indexRelativeToCurrentPage === 0,
        isLastVisible: indexRelativeToCurrentPage === currentPage.rows.length - 1,
        indexRelativeToCurrentPage
      };
      return getRowSpacing((0, _extends2.default)({}, rowEntry, visibility, {
        indexRelativeToCurrentPage: apiRef.current.getRowIndexRelativeToVisibleRows(rowEntry.id)
      }));
    } : undefined, [apiRef, getRowSpacing, currentPage.rows, currentPage.rowIdToIndexMap]),
    applyRowHeight: (0, _useEventCallback.default)((entry, row) => apiRef.current.unstable_applyPipeProcessors('rowHeight', entry, row)),
    virtualizeColumnsWithAutoRowHeight: rootProps.virtualizeColumnsWithAutoRowHeight,
    focusedVirtualCell: (0, _useEventCallback.default)(() => (0, _gridFocusedVirtualCellSelector.gridFocusedVirtualCellSelector)(apiRef)),
    resizeThrottleMs: rootProps.resizeThrottleMs,
    onResize: (0, _useEventCallback.default)(size => apiRef.current.publishEvent('resize', size)),
    onWheel: (0, _useEventCallback.default)(event => {
      apiRef.current.publishEvent('virtualScrollerWheel', {}, event);
    }),
    onTouchMove: (0, _useEventCallback.default)(event => {
      apiRef.current.publishEvent('virtualScrollerTouchMove', {}, event);
    }),
    onRenderContextChange: (0, _useEventCallback.default)(nextRenderContext => {
      apiRef.current.publishEvent('renderedRowsIntervalChange', nextRenderContext);
    }),
    onScrollChange: React.useCallback((scrollPosition, nextRenderContext) => {
      apiRef.current.publishEvent('scrollPositionChange', {
        top: scrollPosition.top,
        left: scrollPosition.left,
        renderContext: nextRenderContext
      });
    }, [apiRef]),
    scrollReset,
    renderRow: React.useCallback(params => /*#__PURE__*/(0, _jsxRuntime.jsx)(RowSlot, (0, _extends2.default)({
      row: params.model,
      rowId: params.id,
      index: params.rowIndex,
      selected: isRowSelected(params.id),
      offsetLeft: params.offsetLeft,
      columnsTotalWidth: columnsTotalWidth,
      rowHeight: params.baseRowHeight,
      pinnedColumns: pinnedColumns,
      visibleColumns: visibleColumns,
      firstColumnIndex: params.firstColumnIndex,
      lastColumnIndex: params.lastColumnIndex,
      focusedColumnIndex: params.focusedColumnIndex,
      isFirstVisible: params.isFirstVisible,
      isLastVisible: params.isLastVisible,
      isNotVisible: params.isVirtualFocusRow,
      showBottomBorder: params.showBottomBorder,
      scrollbarWidth: verticalScrollbarWidth,
      gridHasFiller: hasFiller
    }, rowSlotProps), params.id), [columnsTotalWidth, hasFiller, isRowSelected, pinnedColumns, RowSlot, rowSlotProps, verticalScrollbarWidth, visibleColumns]),
    renderInfiniteLoadingTrigger: React.useCallback(id => apiRef.current.getInfiniteLoadingTriggerElement?.({
      lastRowId: id
    }), [apiRef])
  });

  // HACK: Keep the grid's store in sync with the virtualizer store. We set up the
  // subscription in the render phase rather than in an effect because other grid
  // initialization code runs between those two moments.
  //
  // TODO(v9): Remove this
  (0, _useFirstRender.useFirstRender)(() => {
    apiRef.current.store.state.dimensions = addGridDimensions(virtualizer.store.state.dimensions, headerHeight, groupHeaderHeight, headerFilterHeight, headersTotalHeight);
    apiRef.current.store.state.rowsMeta = virtualizer.store.state.rowsMeta;
    apiRef.current.store.state.virtualization = virtualizer.store.state.virtualization;
  });
  (0, _store.useStoreEffect)(virtualizer.store, _xVirtualizer.Dimensions.selectors.dimensions, (_, dimensions) => {
    if (!dimensions.isReady) {
      return;
    }
    apiRef.current.setState(gridState => (0, _extends2.default)({}, gridState, {
      dimensions: addGridDimensions(dimensions, headerHeight, groupHeaderHeight, headerFilterHeight, headersTotalHeight)
    }));
  });
  (0, _store.useStoreEffect)(virtualizer.store, _xVirtualizer.Dimensions.selectors.rowsMeta, (_, rowsMeta) => {
    if (rowsMeta !== apiRef.current.state.rowsMeta) {
      apiRef.current.setState(gridState => (0, _extends2.default)({}, gridState, {
        rowsMeta
      }));
    }
  });
  (0, _store.useStoreEffect)(virtualizer.store, _xVirtualizer.Virtualization.selectors.store, (_, virtualization) => {
    if (virtualization.renderContext === _xVirtualizer.EMPTY_RENDER_CONTEXT) {
      return;
    }
    if (virtualization !== apiRef.current.state.virtualization) {
      apiRef.current.setState(gridState => (0, _extends2.default)({}, gridState, {
        virtualization
      }));
    }
  });
  apiRef.current.register('private', {
    virtualizer
  });
  (0, _useGridRowsMeta.useGridRowsMeta)(apiRef, rootProps);
  return virtualizer;
}