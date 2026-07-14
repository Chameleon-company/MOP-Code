"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.gridRowReorderStateSelector = exports.gridRowDropTargetSelector = exports.gridRowDropTargetRowIdSelector = exports.gridRowDropPositionSelector = exports.gridIsRowDragActiveSelector = exports.gridDraggedRowIdSelector = void 0;
var _createSelector = require("../../../utils/createSelector");
const gridRowReorderStateSelector = exports.gridRowReorderStateSelector = (0, _createSelector.createRootSelector)(state => state.rowReorder);
const gridIsRowDragActiveSelector = exports.gridIsRowDragActiveSelector = (0, _createSelector.createSelector)(gridRowReorderStateSelector, rowReorder => rowReorder?.isActive ?? false);

// Selector for the entire drop target state
const gridRowDropTargetSelector = exports.gridRowDropTargetSelector = (0, _createSelector.createSelector)(gridRowReorderStateSelector, rowReorder => rowReorder?.dropTarget ?? {
  rowId: null,
  position: null
});
const gridRowDropTargetRowIdSelector = exports.gridRowDropTargetRowIdSelector = (0, _createSelector.createSelector)(gridRowDropTargetSelector, dropTarget => dropTarget.rowId ?? null);

// Selector for a specific row's drop position
const gridRowDropPositionSelector = exports.gridRowDropPositionSelector = (0, _createSelector.createSelector)(gridRowDropTargetSelector, (dropTarget, rowId) => {
  if (dropTarget.rowId === rowId) {
    return dropTarget.position;
  }
  return null;
});

// Selector for the dragged row ID
const gridDraggedRowIdSelector = exports.gridDraggedRowIdSelector = (0, _createSelector.createSelector)(gridRowReorderStateSelector, rowReorder => rowReorder?.draggedRowId ?? null);