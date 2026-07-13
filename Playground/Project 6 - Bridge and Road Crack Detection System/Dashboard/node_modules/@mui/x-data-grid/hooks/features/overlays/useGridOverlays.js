"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useGridOverlays = void 0;
var _utils = require("../../utils");
var _filter = require("../filter");
var _rows = require("../rows");
var _gridRowsSelector = require("../rows/gridRowsSelector");
var _columns = require("../columns");
var _pivoting = require("../pivoting");
/**
 * Uses the grid state to determine which overlay to display.
 * Returns the active overlay type and the active loading overlay variant.
 */
const useGridOverlays = (apiRef, props) => {
  const totalRowCount = (0, _utils.useGridSelector)(apiRef, _rows.gridRowCountSelector);
  const visibleRowCount = (0, _utils.useGridSelector)(apiRef, _filter.gridExpandedRowCountSelector);
  const pinnedRowsCount = (0, _utils.useGridSelector)(apiRef, _gridRowsSelector.gridPinnedRowsCountSelector);
  const visibleColumns = (0, _utils.useGridSelector)(apiRef, _columns.gridVisibleColumnDefinitionsSelector);
  const noRows = totalRowCount === 0 && pinnedRowsCount === 0;
  const loading = (0, _utils.useGridSelector)(apiRef, _rows.gridRowsLoadingSelector);
  const pivotActive = (0, _utils.useGridSelector)(apiRef, _pivoting.gridPivotActiveSelector);
  const showNoRowsOverlay = !loading && noRows;
  const showNoResultsOverlay = !loading && totalRowCount > 0 && visibleRowCount === 0;
  const showNoColumnsOverlay = !loading && visibleColumns.length === 0;
  const showEmptyPivotOverlay = showNoRowsOverlay && pivotActive;
  let overlayType = null;
  let loadingOverlayVariant = null;
  if (showNoRowsOverlay) {
    overlayType = 'noRowsOverlay';
  }
  if (showNoColumnsOverlay) {
    overlayType = 'noColumnsOverlay';
  }
  if (showEmptyPivotOverlay) {
    overlayType = 'emptyPivotOverlay';
  }
  if (showNoResultsOverlay) {
    overlayType = 'noResultsOverlay';
  }
  if (loading) {
    overlayType = 'loadingOverlay';
    loadingOverlayVariant = props.slotProps?.loadingOverlay?.[noRows ? 'noRowsVariant' : 'variant'] ?? (noRows ? 'skeleton' : 'linear-progress');
  }
  return {
    overlayType: overlayType,
    loadingOverlayVariant
  };
};
exports.useGridOverlays = useGridOverlays;