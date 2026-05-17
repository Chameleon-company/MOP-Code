"use strict";
'use client';

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useGridRowSpanning = exports.rowSpanningStateInitializer = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var React = _interopRequireWildcard(require("react"));
var _features = require("@mui/x-virtualizer/features");
var _gridColumnsSelector = require("../columns/gridColumnsSelector");
var _useGridVisibleRows = require("../../utils/useGridVisibleRows");
var _gridVirtualizationSelectors = require("../virtualization/gridVirtualizationSelectors");
var _gridRowSpanningUtils = require("./gridRowSpanningUtils");
var _useGridEvent = require("../../utils/useGridEvent");
var _utils = require("../../../utils/utils");
var _useRunOncePerLoop = require("../../utils/useRunOncePerLoop");
const EMPTY_CACHES = {
  spannedCells: {},
  hiddenCells: {},
  hiddenCellOriginMap: {}
};
const EMPTY_RANGE = {
  firstRowIndex: 0,
  lastRowIndex: 0
};
const EMPTY_STATE = {
  caches: EMPTY_CACHES,
  processedRange: EMPTY_RANGE
};
const computeRowSpanningState = (apiRef, colDefs, visibleRows, range, rangeToProcess, resetState) => {
  const virtualizer = apiRef.current.virtualizer;
  const previousState = resetState ? EMPTY_STATE : _features.Rowspan.selectors.state(virtualizer.store.state);
  const spannedCells = (0, _extends2.default)({}, previousState.caches.spannedCells);
  const hiddenCells = (0, _extends2.default)({}, previousState.caches.hiddenCells);
  const hiddenCellOriginMap = (0, _extends2.default)({}, previousState.caches.hiddenCellOriginMap);
  const processedRange = {
    firstRowIndex: Math.min(previousState.processedRange.firstRowIndex, rangeToProcess.firstRowIndex),
    lastRowIndex: Math.max(previousState.processedRange.lastRowIndex, rangeToProcess.lastRowIndex)
  };
  colDefs.forEach((colDef, columnIndex) => {
    for (let index = rangeToProcess.firstRowIndex; index < rangeToProcess.lastRowIndex; index += 1) {
      const row = visibleRows[index];
      if (hiddenCells[row.id]?.[columnIndex]) {
        continue;
      }
      const cellValue = (0, _gridRowSpanningUtils.getCellValue)(row.model, colDef, apiRef);
      if (cellValue == null) {
        continue;
      }
      let spannedRowId = row.id;
      let spannedRowIndex = index;
      let rowSpan = 0;

      // For first index, also scan in the previous rows to handle the reset state case e.g by sorting
      const backwardsHiddenCells = [];
      if (index === rangeToProcess.firstRowIndex) {
        let prevIndex = index - 1;
        let prevRowEntry = visibleRows[prevIndex];
        while (prevIndex >= range.firstRowIndex && prevRowEntry && (0, _gridRowSpanningUtils.getCellValue)(prevRowEntry.model, colDef, apiRef) === cellValue) {
          const currentRow = visibleRows[prevIndex + 1];
          if (hiddenCells[currentRow.id]) {
            hiddenCells[currentRow.id][columnIndex] = true;
          } else {
            hiddenCells[currentRow.id] = {
              [columnIndex]: true
            };
          }
          backwardsHiddenCells.push(index);
          rowSpan += 1;
          spannedRowId = prevRowEntry.id;
          spannedRowIndex = prevIndex;
          prevIndex -= 1;
          prevRowEntry = visibleRows[prevIndex];
        }
      }
      backwardsHiddenCells.forEach(hiddenCellIndex => {
        if (hiddenCellOriginMap[hiddenCellIndex]) {
          hiddenCellOriginMap[hiddenCellIndex][columnIndex] = spannedRowIndex;
        } else {
          hiddenCellOriginMap[hiddenCellIndex] = {
            [columnIndex]: spannedRowIndex
          };
        }
      });

      // Scan the next rows
      let relativeIndex = index + 1;
      while (relativeIndex <= range.lastRowIndex && visibleRows[relativeIndex] && (0, _gridRowSpanningUtils.getCellValue)(visibleRows[relativeIndex].model, colDef, apiRef) === cellValue) {
        const currentRow = visibleRows[relativeIndex];
        if (hiddenCells[currentRow.id]) {
          hiddenCells[currentRow.id][columnIndex] = true;
        } else {
          hiddenCells[currentRow.id] = {
            [columnIndex]: true
          };
        }
        if (hiddenCellOriginMap[relativeIndex]) {
          hiddenCellOriginMap[relativeIndex][columnIndex] = spannedRowIndex;
        } else {
          hiddenCellOriginMap[relativeIndex] = {
            [columnIndex]: spannedRowIndex
          };
        }
        relativeIndex += 1;
        rowSpan += 1;
      }
      if (rowSpan > 0) {
        if (spannedCells[spannedRowId]) {
          spannedCells[spannedRowId][columnIndex] = rowSpan + 1;
        } else {
          spannedCells[spannedRowId] = {
            [columnIndex]: rowSpan + 1
          };
        }
      }
    }
  });
  return {
    caches: {
      spannedCells,
      hiddenCells,
      hiddenCellOriginMap
    },
    processedRange
  };
};

/**
 * @requires columnsStateInitializer (method) - should be initialized before
 * @requires rowsStateInitializer (method) - should be initialized before
 * @requires filterStateInitializer (method) - should be initialized before
 */
const rowSpanningStateInitializer = state => {
  return (0, _extends2.default)({}, state, {
    rowSpanning: EMPTY_STATE
  });
};
exports.rowSpanningStateInitializer = rowSpanningStateInitializer;
const useGridRowSpanning = (apiRef, props) => {
  const updateRowSpanningState = React.useCallback((renderContext, resetState = false) => {
    const store = apiRef.current.virtualizer.store;
    const {
      range,
      rows: visibleRows
    } = (0, _useGridVisibleRows.getVisibleRows)(apiRef);
    if (resetState) {
      store.set('rowSpanning', EMPTY_STATE);
    }
    if (range === null || !(0, _gridRowSpanningUtils.isRowContextInitialized)(renderContext)) {
      return;
    }
    const previousState = resetState ? EMPTY_STATE : _features.Rowspan.selectors.state(store.state);
    const rangeToProcess = (0, _gridRowSpanningUtils.getUnprocessedRange)({
      firstRowIndex: renderContext.firstRowIndex,
      lastRowIndex: Math.min(renderContext.lastRowIndex, range.lastRowIndex - range.firstRowIndex + 1)
    }, previousState.processedRange);
    if (rangeToProcess === null) {
      return;
    }
    const colDefs = (0, _gridColumnsSelector.gridVisibleColumnDefinitionsSelector)(apiRef);
    const newState = computeRowSpanningState(apiRef, colDefs, visibleRows, range, rangeToProcess, resetState);
    const newSpannedCellsCount = Object.keys(newState.caches.spannedCells).length;
    const newHiddenCellsCount = Object.keys(newState.caches.hiddenCells).length;
    const previousSpannedCellsCount = Object.keys(previousState.caches.spannedCells).length;
    const previousHiddenCellsCount = Object.keys(previousState.caches.hiddenCells).length;
    const shouldUpdateState = resetState || newSpannedCellsCount !== previousSpannedCellsCount || newHiddenCellsCount !== previousHiddenCellsCount;
    const hasNoSpannedCells = newSpannedCellsCount === 0 && previousSpannedCellsCount === 0;
    if (!shouldUpdateState || hasNoSpannedCells) {
      return;
    }
    store.set('rowSpanning', newState);
  }, [apiRef]);

  // Reset events trigger a full re-computation of the row spanning state:
  // - The `rowSpanning` prop is updated (feature flag)
  // - The filtering is applied
  // - The sorting is applied
  // - The `paginationModel` is updated
  // - The rows are updated
  const {
    schedule: deferredUpdateRowSpanningState,
    cancel
  } = (0, _useRunOncePerLoop.useRunOncePerLoop)(updateRowSpanningState);
  const resetRowSpanningState = React.useCallback(() => {
    const renderContext = (0, _gridVirtualizationSelectors.gridRenderContextSelector)(apiRef);
    if (!(0, _gridRowSpanningUtils.isRowContextInitialized)(renderContext)) {
      return;
    }
    deferredUpdateRowSpanningState(renderContext, true);
  }, [apiRef, deferredUpdateRowSpanningState]);
  (0, _useGridEvent.useGridEvent)(apiRef, 'renderedRowsIntervalChange', (0, _utils.runIf)(props.rowSpanning, renderContext => {
    const didHavePendingReset = cancel();
    updateRowSpanningState(renderContext, didHavePendingReset);
  }));
  (0, _useGridEvent.useGridEvent)(apiRef, 'sortedRowsSet', (0, _utils.runIf)(props.rowSpanning, resetRowSpanningState));
  (0, _useGridEvent.useGridEvent)(apiRef, 'paginationModelChange', (0, _utils.runIf)(props.rowSpanning, resetRowSpanningState));
  (0, _useGridEvent.useGridEvent)(apiRef, 'filteredRowsSet', (0, _utils.runIf)(props.rowSpanning, resetRowSpanningState));
  (0, _useGridEvent.useGridEvent)(apiRef, 'columnsChange', (0, _utils.runIf)(props.rowSpanning, resetRowSpanningState));
  (0, _useGridEvent.useGridEvent)(apiRef, 'rowExpansionChange', (0, _utils.runIf)(props.rowSpanning, resetRowSpanningState));
  React.useEffect(() => {
    const store = apiRef.current.virtualizer?.store;
    if (!store) {
      return;
    }
    if (!props.rowSpanning) {
      if (store.state.rowSpanning !== EMPTY_STATE) {
        store.set('rowSpanning', EMPTY_STATE);
      }
    } else if (store.state.rowSpanning === EMPTY_STATE) {
      updateRowSpanningState((0, _gridVirtualizationSelectors.gridRenderContextSelector)(apiRef));
    }
  }, [apiRef, props.rowSpanning, updateRowSpanningState]);
};
exports.useGridRowSpanning = useGridRowSpanning;