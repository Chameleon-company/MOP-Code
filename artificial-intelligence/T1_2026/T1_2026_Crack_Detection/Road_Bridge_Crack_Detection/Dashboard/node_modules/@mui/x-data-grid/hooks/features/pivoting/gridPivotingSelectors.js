"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.gridPivotInitialColumnsSelector = exports.gridPivotActiveSelector = void 0;
var _createSelector = require("../../../utils/createSelector");
const gridPivotingStateSelector = (0, _createSelector.createRootSelector)(
// @ts-ignore
state => state.pivoting);
const gridPivotActiveSelector = exports.gridPivotActiveSelector = (0, _createSelector.createSelector)(gridPivotingStateSelector, pivoting => pivoting?.active);
const emptyColumns = new Map();
const gridPivotInitialColumnsSelector = exports.gridPivotInitialColumnsSelector = (0, _createSelector.createSelector)(gridPivotingStateSelector, pivoting => pivoting?.initialColumns || emptyColumns);