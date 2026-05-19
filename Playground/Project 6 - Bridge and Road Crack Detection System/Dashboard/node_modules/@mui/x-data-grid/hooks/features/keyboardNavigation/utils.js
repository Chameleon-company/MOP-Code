"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.findNonRowSpannedCell = findNonRowSpannedCell;
exports.getRightColumnIndex = exports.getLeftColumnIndex = void 0;
var _xVirtualizer = require("@mui/x-virtualizer");
var _gridFilterSelector = require("../filter/gridFilterSelector");
const getLeftColumnIndex = ({
  currentColIndex,
  firstColIndex,
  lastColIndex,
  isRtl
}) => {
  if (isRtl) {
    if (currentColIndex < lastColIndex) {
      return currentColIndex + 1;
    }
  } else if (!isRtl) {
    if (currentColIndex > firstColIndex) {
      return currentColIndex - 1;
    }
  }
  return null;
};
exports.getLeftColumnIndex = getLeftColumnIndex;
const getRightColumnIndex = ({
  currentColIndex,
  firstColIndex,
  lastColIndex,
  isRtl
}) => {
  if (isRtl) {
    if (currentColIndex > firstColIndex) {
      return currentColIndex - 1;
    }
  } else if (!isRtl) {
    if (currentColIndex < lastColIndex) {
      return currentColIndex + 1;
    }
  }
  return null;
};
exports.getRightColumnIndex = getRightColumnIndex;
function findNonRowSpannedCell(apiRef, rowId, colIndex, rowSpanScanDirection) {
  const rowSpanHiddenCells = _xVirtualizer.Rowspan.selectors.hiddenCells(apiRef.current.virtualizer.store.state);
  if (!rowSpanHiddenCells[rowId]?.[colIndex]) {
    return rowId;
  }
  const filteredSortedRowIds = (0, _gridFilterSelector.gridFilteredSortedRowIdsSelector)(apiRef);
  // find closest non row spanned cell in the given `rowSpanScanDirection`
  let nextRowIndex = filteredSortedRowIds.indexOf(rowId) + (rowSpanScanDirection === 'down' ? 1 : -1);
  while (nextRowIndex >= 0 && nextRowIndex < filteredSortedRowIds.length) {
    const nextRowId = filteredSortedRowIds[nextRowIndex];
    if (!rowSpanHiddenCells[nextRowId]?.[colIndex]) {
      return nextRowId;
    }
    nextRowIndex += rowSpanScanDirection === 'down' ? 1 : -1;
  }
  return rowId;
}