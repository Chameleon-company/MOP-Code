"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.findRowsToSelect = exports.findRowsToDeselect = exports.checkboxPropsSelector = exports.ROW_SELECTION_PROPAGATION_DEFAULT = void 0;
exports.isMultipleRowSelectionEnabled = isMultipleRowSelectionEnabled;
var _signature = require("../../../constants/signature");
var _gridRowsUtils = require("../rows/gridRowsUtils");
var _gridFilterSelector = require("../filter/gridFilterSelector");
var _gridSortingSelector = require("../sorting/gridSortingSelector");
var _gridRowSelectionSelector = require("./gridRowSelectionSelector");
var _gridRowsSelector = require("../rows/gridRowsSelector");
var _createSelector = require("../../../utils/createSelector");
var _columns = require("../columns");
var _gridPropsSelectors = require("../../core/gridPropsSelectors");
const ROW_SELECTION_PROPAGATION_DEFAULT = exports.ROW_SELECTION_PROPAGATION_DEFAULT = {
  parents: true,
  descendants: true
};
function getGridRowGroupSelectableDescendants(apiRef, groupId) {
  const rowTree = (0, _gridRowsSelector.gridRowTreeSelector)(apiRef);
  const sortedRowIds = (0, _gridSortingSelector.gridSortedRowIdsSelector)(apiRef);
  const filteredRowsLookup = (0, _gridFilterSelector.gridFilteredRowsLookupSelector)(apiRef);
  const groupNode = rowTree[groupId];
  if (!groupNode || groupNode.type !== 'group') {
    return [];
  }
  const descendants = [];
  const startIndex = sortedRowIds.findIndex(id => id === groupId) + 1;
  for (let index = startIndex; index < sortedRowIds.length && rowTree[sortedRowIds[index]]?.depth > groupNode.depth; index += 1) {
    const id = sortedRowIds[index];
    if (filteredRowsLookup[id] !== false && apiRef.current.isRowSelectable(id)) {
      descendants.push(id);
    }
  }
  return descendants;
}
const checkboxPropsSelector = exports.checkboxPropsSelector = (0, _createSelector.createSelector)(_columns.gridColumnDefinitionsSelector, _gridRowsSelector.gridRowTreeSelector, _gridFilterSelector.gridFilteredRowsLookupSelector, _gridRowSelectionSelector.gridRowSelectionManagerSelector, _gridRowsSelector.gridRowsLookupSelector, _gridPropsSelectors.gridRowSelectableSelector, (columns, rowTree, filteredRowsLookup, rowSelectionManager, rowsLookup, isRowSelectable, {
  groupId,
  autoSelectParents
}) => {
  const groupNode = rowTree[groupId];
  const rowParams = {
    id: groupId,
    row: rowsLookup[groupId],
    columns
  };
  let isSelectable = true;
  if (typeof isRowSelectable === 'function' && rowsLookup[groupId]) {
    isSelectable = isRowSelectable(rowParams);
  }
  if (!groupNode || groupNode.type !== 'group' || rowSelectionManager.has(groupId)) {
    return {
      isIndeterminate: false,
      isChecked: rowSelectionManager.has(groupId),
      isSelectable
    };
  }
  let hasSelectedDescendant = false;
  let hasUnSelectedDescendant = false;
  const traverseDescendants = itemToTraverseId => {
    if (filteredRowsLookup[itemToTraverseId] === false ||
    // Perf: Skip checking the rest of the descendants if we already
    // know that there is a selected and an unselected descendant
    hasSelectedDescendant && hasUnSelectedDescendant) {
      return;
    }
    const node = rowTree[itemToTraverseId];
    if (node?.type === 'group') {
      node.children.forEach(traverseDescendants);
    }
    // Check if row is selectable before considering it for parent selection state
    const descendantRowParams = {
      id: itemToTraverseId,
      row: rowsLookup[itemToTraverseId],
      columns
    };
    const rowIsSelectable = typeof isRowSelectable === 'function' && rowsLookup[itemToTraverseId] ? isRowSelectable(descendantRowParams) : true;

    // Only consider selectable rows when determining parent selection state
    if (rowIsSelectable) {
      if (rowSelectionManager.has(itemToTraverseId)) {
        hasSelectedDescendant = true;
      } else {
        hasUnSelectedDescendant = true;
      }
    }
  };
  traverseDescendants(groupId);
  return {
    isIndeterminate: hasSelectedDescendant && hasUnSelectedDescendant,
    isChecked: autoSelectParents ? hasSelectedDescendant && !hasUnSelectedDescendant : false,
    isSelectable
  };
});
function isMultipleRowSelectionEnabled(props) {
  if (props.signature === _signature.GridSignature.DataGrid) {
    // DataGrid Community has multiple row selection enabled only if checkbox selection is enabled.
    return props.checkboxSelection && props.disableMultipleRowSelection !== true;
  }
  return !props.disableMultipleRowSelection;
}
const getRowNodeParents = (tree, id) => {
  const parents = [];
  let parent = id;
  while (parent != null && parent !== _gridRowsUtils.GRID_ROOT_GROUP_ID) {
    const node = tree[parent];
    if (!node) {
      return parents;
    }
    parents.push(parent);
    parent = node.parent;
  }
  return parents;
};
const getFilteredRowNodeSiblings = (tree, filteredRows, id) => {
  const node = tree[id];
  if (!node) {
    return [];
  }
  const parent = node.parent;
  if (parent == null) {
    return [];
  }
  const parentNode = tree[parent];
  return parentNode.children.filter(childId => childId !== id && filteredRows[childId] !== false);
};
const findRowsToSelect = (apiRef, tree, selectedRow, autoSelectDescendants, autoSelectParents, addRow, rowSelectionManager = (0, _gridRowSelectionSelector.gridRowSelectionManagerSelector)(apiRef)) => {
  const filteredRows = (0, _gridFilterSelector.gridFilteredRowsLookupSelector)(apiRef);
  const selectedDescendants = new Set([]);
  if (!autoSelectDescendants && !autoSelectParents || filteredRows[selectedRow] === false) {
    return;
  }
  if (autoSelectDescendants) {
    const rowNode = tree[selectedRow];
    if (rowNode?.type === 'group') {
      const descendants = getGridRowGroupSelectableDescendants(apiRef, selectedRow);
      descendants.forEach(rowId => {
        addRow(rowId);
        selectedDescendants.add(rowId);
      });
    }
  }
  if (autoSelectParents) {
    const checkAllDescendantsSelected = rowId => {
      const node = tree[rowId];
      if (!node) {
        return false;
      }
      // For non-group nodes, check if it's selected or if it's non-selectable
      if (node.type !== 'group') {
        // If the row is selectable, it must be selected
        if (apiRef.current.isRowSelectable(rowId)) {
          return rowSelectionManager.has(rowId) || selectedDescendants.has(rowId);
        }
        // Non-selectable rows don't affect parent selection
        return true;
      }
      // For group nodes, check if it's selected or all its children are selected
      if (rowSelectionManager.has(rowId) || selectedDescendants.has(rowId)) {
        return true;
      }
      // Recursively check all children
      return node.children.every(checkAllDescendantsSelected);
    };
    const traverseParents = rowId => {
      const siblings = getFilteredRowNodeSiblings(tree, filteredRows, rowId);
      // Check if all selectable siblings are selected
      const allSelectableSiblingsSelected = siblings.every(siblingId => {
        // Non-selectable siblings don't affect parent selection
        if (!apiRef.current.isRowSelectable(siblingId)) {
          return true;
        }
        return checkAllDescendantsSelected(siblingId);
      });
      if (siblings.length === 0 || allSelectableSiblingsSelected) {
        const rowNode = tree[rowId];
        const parent = rowNode?.parent;
        if (parent != null && parent !== _gridRowsUtils.GRID_ROOT_GROUP_ID && apiRef.current.isRowSelectable(parent)) {
          addRow(parent);
          selectedDescendants.add(parent);
          traverseParents(parent);
        }
      }
    };
    // For root level rows, we don't need to traverse parents
    const rowNode = tree[selectedRow];
    if (!rowNode || rowNode.parent === _gridRowsUtils.GRID_ROOT_GROUP_ID) {
      return;
    }
    traverseParents(selectedRow);
  }
};
exports.findRowsToSelect = findRowsToSelect;
const findRowsToDeselect = (apiRef, tree, deselectedRow, autoSelectDescendants, autoSelectParents, removeRow) => {
  const rowSelectionManager = (0, _gridRowSelectionSelector.gridRowSelectionManagerSelector)(apiRef);
  if (!autoSelectParents && !autoSelectDescendants) {
    return;
  }
  if (autoSelectParents) {
    const allParents = getRowNodeParents(tree, deselectedRow);
    allParents.forEach(parent => {
      const isSelected = rowSelectionManager.has(parent);
      if (isSelected) {
        removeRow(parent);
      }
    });
  }
  if (autoSelectDescendants) {
    const rowNode = tree[deselectedRow];
    if (rowNode?.type === 'group') {
      const descendants = getGridRowGroupSelectableDescendants(apiRef, deselectedRow);
      descendants.forEach(descendant => {
        removeRow(descendant);
      });
    }
  }
};
exports.findRowsToDeselect = findRowsToDeselect;