"use strict";
'use client';

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.GridFooterCell = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var _objectWithoutPropertiesLoose2 = _interopRequireDefault(require("@babel/runtime/helpers/objectWithoutPropertiesLoose"));
var React = _interopRequireWildcard(require("react"));
var _composeClasses = _interopRequireDefault(require("@mui/utils/composeClasses"));
var _styles = require("@mui/material/styles");
var _cssVariables = require("../../constants/cssVariables");
var _useGridRootProps = require("../../hooks/utils/useGridRootProps");
var _gridClasses = require("../../constants/gridClasses");
var _jsxRuntime = require("react/jsx-runtime");
const _excluded = ["formattedValue", "colDef", "cellMode", "row", "api", "id", "value", "rowNode", "field", "hasFocus", "tabIndex", "isEditable"];
const GridFooterCellRoot = (0, _styles.styled)('div', {
  name: 'MuiDataGrid',
  slot: 'FooterCell'
})({
  fontWeight: _cssVariables.vars.typography.fontWeight.medium,
  color: _cssVariables.vars.colors.foreground.accent
});
const useUtilityClasses = ownerState => {
  const {
    classes
  } = ownerState;
  const slots = {
    root: ['footerCell']
  };
  return (0, _composeClasses.default)(slots, _gridClasses.getDataGridUtilityClass, classes);
};
function GridFooterCellRaw(props) {
  const {
      formattedValue
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded);
  const rootProps = (0, _useGridRootProps.useGridRootProps)();
  const ownerState = {
    classes: rootProps.classes
  };
  const classes = useUtilityClasses(ownerState);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(GridFooterCellRoot, (0, _extends2.default)({
    ownerState: ownerState,
    className: classes.root
  }, other, {
    children: formattedValue
  }));
}
const GridFooterCell = exports.GridFooterCell = /*#__PURE__*/React.memo(GridFooterCellRaw);
if (process.env.NODE_ENV !== "production") GridFooterCell.displayName = "GridFooterCell";