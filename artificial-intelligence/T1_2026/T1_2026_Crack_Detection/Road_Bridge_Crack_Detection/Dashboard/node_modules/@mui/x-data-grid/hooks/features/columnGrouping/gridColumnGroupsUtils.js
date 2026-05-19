"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.unwrapGroupingColumnModel = exports.getColumnGroupsHeaderStructure = exports.createGroupLookup = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var _objectWithoutPropertiesLoose2 = _interopRequireDefault(require("@babel/runtime/helpers/objectWithoutPropertiesLoose"));
var _gridColumnGrouping = require("../../../models/gridColumnGrouping");
const _excluded = ["groupId", "children"];
const createGroupLookup = columnGroupingModel => {
  const groupLookup = {};
  for (let i = 0; i < columnGroupingModel.length; i += 1) {
    const node = columnGroupingModel[i];
    if ((0, _gridColumnGrouping.isLeaf)(node)) {
      continue;
    }
    const {
        groupId,
        children
      } = node,
      other = (0, _objectWithoutPropertiesLoose2.default)(node, _excluded);
    if (!groupId) {
      throw new Error('MUI X: An element of the columnGroupingModel does not have either `field` or `groupId`.');
    }
    if (process.env.NODE_ENV !== 'production' && !children) {
      console.warn(`MUI X: group groupId=${groupId} has no children.`);
    }
    const groupParam = (0, _extends2.default)({}, other, {
      groupId
    });
    const subTreeLookup = createGroupLookup(children);
    if (subTreeLookup[groupId] !== undefined || groupLookup[groupId] !== undefined) {
      throw new Error(`MUI X: The groupId ${groupId} is used multiple times in the columnGroupingModel.`);
    }
    Object.assign(groupLookup, subTreeLookup);
    groupLookup[groupId] = groupParam;
  }
  return groupLookup;
};

// This is the recurrence function that help writing `unwrapGroupingColumnModel()`
exports.createGroupLookup = createGroupLookup;
const recurrentUnwrapGroupingColumnModel = (columnGroupNode, parents, unwrappedGroupingModelToComplete) => {
  if ((0, _gridColumnGrouping.isLeaf)(columnGroupNode)) {
    if (unwrappedGroupingModelToComplete[columnGroupNode.field] !== undefined) {
      throw new Error([`MUI X: columnGroupingModel contains duplicated field`, `column field ${columnGroupNode.field} occurs two times in the grouping model:`, `- ${unwrappedGroupingModelToComplete[columnGroupNode.field].join(' > ')}`, `- ${parents.join(' > ')}`].join('\n'));
    }
    unwrappedGroupingModelToComplete[columnGroupNode.field] = parents;
    return;
  }
  const {
    groupId,
    children
  } = columnGroupNode;
  children.forEach(child => {
    recurrentUnwrapGroupingColumnModel(child, [...parents, groupId], unwrappedGroupingModelToComplete);
  });
};

/**
 * This is a function that provide for each column the array of its parents.
 * Parents are ordered from the root to the leaf.
 * @param columnGroupingModel The model such as provided in DataGrid props
 * @returns An object `{[field]: groupIds}` where `groupIds` is the parents of the column `field`
 */
const unwrapGroupingColumnModel = columnGroupingModel => {
  if (!columnGroupingModel) {
    return {};
  }
  const unwrappedSubTree = {};
  columnGroupingModel.forEach(columnGroupNode => {
    recurrentUnwrapGroupingColumnModel(columnGroupNode, [], unwrappedSubTree);
  });
  return unwrappedSubTree;
};
exports.unwrapGroupingColumnModel = unwrapGroupingColumnModel;
const getColumnGroupsHeaderStructure = (orderedColumns, unwrappedGroupingModel, pinnedFields) => {
  const getParents = field => unwrappedGroupingModel[field] ?? [];
  const groupingHeaderStructure = [];
  const maxDepth = Math.max(0, ...orderedColumns.map(field => getParents(field).length));
  const haveSameParents = (field1, field2, depth) => {
    const a = getParents(field1);
    const b = getParents(field2);
    for (let i = 0; i <= depth; i += 1) {
      if (a[i] !== b[i]) {
        return false;
      }
    }
    return true;
  };
  const haveDifferentContainers = (field1, field2) => {
    const left = pinnedFields?.left;
    const right = pinnedFields?.right;
    const inLeft1 = !!left?.includes(field1);
    const inLeft2 = !!left?.includes(field2);
    const inRight1 = !!right?.includes(field1);
    const inRight2 = !!right?.includes(field2);
    return inLeft1 !== inLeft2 || inRight1 !== inRight2;
  };
  for (let depth = 0; depth < maxDepth; depth += 1) {
    const depthStructure = [];
    for (let i = 0; i < orderedColumns.length; i += 1) {
      const field = orderedColumns[i];
      const groupId = getParents(field)[depth] ?? null;
      if (depthStructure.length === 0) {
        depthStructure.push({
          columnFields: [field],
          groupId
        });
        continue;
      }
      const lastGroup = depthStructure[depthStructure.length - 1];
      const prevField = lastGroup.columnFields[lastGroup.columnFields.length - 1];
      if (lastGroup.groupId !== groupId || !haveSameParents(prevField, field, depth) || haveDifferentContainers(prevField, field)) {
        depthStructure.push({
          columnFields: [field],
          groupId
        });
      } else {
        lastGroup.columnFields.push(field);
      }
    }
    groupingHeaderStructure.push(depthStructure);
  }
  return groupingHeaderStructure;
};
exports.getColumnGroupsHeaderStructure = getColumnGroupsHeaderStructure;