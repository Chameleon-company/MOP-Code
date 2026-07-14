"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.GRID_LONG_TEXT_COL_DEF = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var _gridStringColDef = require("./gridStringColDef");
var _GridLongTextCell = require("../components/cell/GridLongTextCell");
var _GridEditLongTextCell = require("../components/cell/GridEditLongTextCell");
const GRID_LONG_TEXT_COL_DEF = exports.GRID_LONG_TEXT_COL_DEF = (0, _extends2.default)({}, _gridStringColDef.GRID_STRING_COL_DEF, {
  type: 'longText',
  display: 'flex',
  renderCell: _GridLongTextCell.renderLongTextCell,
  renderEditCell: _GridEditLongTextCell.renderEditLongTextCell
});