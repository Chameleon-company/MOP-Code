"use strict";

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useGridRowsOverridableMethods = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var React = _interopRequireWildcard(require("react"));
var _gridRowsSelector = require("./gridRowsSelector");
var _gridFilterSelector = require("../filter/gridFilterSelector");
var _gridRowsUtils = require("./gridRowsUtils");
const useGridRowsOverridableMethods = apiRef => {
  const setRowPosition = React.useCallback((sourceRowId, targetRowId, position) => {
    const sourceNode = (0, _gridRowsSelector.gridRowNodeSelector)(apiRef, sourceRowId);
    const targetNode = (0, _gridRowsSelector.gridRowNodeSelector)(apiRef, targetRowId);
    if (!sourceNode) {
      throw new Error(`MUI X: No row with id #${sourceRowId} found.`);
    }
    if (!targetNode) {
      throw new Error(`MUI X: No row with id #${targetRowId} found.`);
    }
    if (sourceNode.type !== 'leaf') {
      throw new Error(`MUI X: The row reordering does not support reordering of footer or grouping rows.`);
    }
    if (position === 'inside') {
      throw new Error(`MUI X: The 'inside' position is only supported for tree data. Use 'above' or 'below' for flat data.`);
    }

    // Get the target index from the targetRowId using the lookup selector
    const sortedFilteredRowIndexLookup = (0, _gridFilterSelector.gridExpandedSortedRowIndexLookupSelector)(apiRef);
    const targetRowIndexUnadjusted = sortedFilteredRowIndexLookup[targetRowId];
    if (targetRowIndexUnadjusted === undefined) {
      throw new Error(`MUI X: Target row with id #${targetRowId} not found in current view.`);
    }
    const sourceRowIndex = sortedFilteredRowIndexLookup[sourceRowId];
    if (sourceRowIndex === undefined) {
      throw new Error(`MUI X: Source row with id #${sourceRowId} not found in current view.`);
    }
    const dragDirection = targetRowIndexUnadjusted < sourceRowIndex ? 'up' : 'down';
    let targetRowIndex;
    if (dragDirection === 'up') {
      targetRowIndex = position === 'above' ? targetRowIndexUnadjusted : targetRowIndexUnadjusted + 1;
    } else {
      targetRowIndex = position === 'above' ? targetRowIndexUnadjusted - 1 : targetRowIndexUnadjusted;
    }
    if (targetRowIndex === sourceRowIndex) {
      return;
    }
    apiRef.current.setState(state => {
      const group = (0, _gridRowsSelector.gridRowTreeSelector)(apiRef)[_gridRowsUtils.GRID_ROOT_GROUP_ID];
      const allRows = group.children;
      const updatedRows = [...allRows];
      updatedRows.splice(targetRowIndex, 0, updatedRows.splice(sourceRowIndex, 1)[0]);
      return (0, _extends2.default)({}, state, {
        rows: (0, _extends2.default)({}, state.rows, {
          tree: (0, _extends2.default)({}, state.rows.tree, {
            [_gridRowsUtils.GRID_ROOT_GROUP_ID]: (0, _extends2.default)({}, group, {
              children: updatedRows
            })
          })
        })
      });
    });
    apiRef.current.publishEvent('rowsSet');
  }, [apiRef]);
  const setRowIndex = React.useCallback((rowId, targetIndex) => {
    const node = (0, _gridRowsSelector.gridRowNodeSelector)(apiRef, rowId);
    if (!node) {
      throw new Error(`MUI X: No row with id #${rowId} found.`);
    }
    if (node.type !== 'leaf') {
      throw new Error(`MUI X: The row reordering does not support reordering of footer or grouping rows.`);
    }
    apiRef.current.setState(state => {
      const group = (0, _gridRowsSelector.gridRowTreeSelector)(apiRef)[_gridRowsUtils.GRID_ROOT_GROUP_ID];
      const allRows = group.children;
      const oldIndex = allRows.findIndex(row => row === rowId);
      if (oldIndex === -1 || oldIndex === targetIndex) {
        return state;
      }
      const updatedRows = [...allRows];
      updatedRows.splice(targetIndex, 0, updatedRows.splice(oldIndex, 1)[0]);
      return (0, _extends2.default)({}, state, {
        rows: (0, _extends2.default)({}, state.rows, {
          tree: (0, _extends2.default)({}, state.rows.tree, {
            [_gridRowsUtils.GRID_ROOT_GROUP_ID]: (0, _extends2.default)({}, group, {
              children: updatedRows
            })
          })
        })
      });
    });
    apiRef.current.publishEvent('rowsSet');
  }, [apiRef]);
  return {
    setRowIndex,
    setRowPosition
  };
};
exports.useGridRowsOverridableMethods = useGridRowsOverridableMethods;