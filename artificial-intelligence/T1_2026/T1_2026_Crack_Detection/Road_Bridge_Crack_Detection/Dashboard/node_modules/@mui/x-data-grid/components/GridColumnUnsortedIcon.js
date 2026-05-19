"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.GridColumnUnsortedIcon = GridColumnUnsortedIcon;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var _objectWithoutPropertiesLoose2 = _interopRequireDefault(require("@babel/runtime/helpers/objectWithoutPropertiesLoose"));
var React = _interopRequireWildcard(require("react"));
var _propTypes = _interopRequireDefault(require("prop-types"));
var _useGridRootProps = require("../hooks/utils/useGridRootProps");
var _jsxRuntime = require("react/jsx-runtime");
const _excluded = ["sortingOrder"];
function GridColumnUnsortedIcon(props) {
  const {
      sortingOrder
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded);
  const {
    slots
  } = (0, _useGridRootProps.useGridRootProps)();
  const [nextSortDirection] = sortingOrder;
  const Icon = nextSortDirection === 'asc' ? slots.columnSortedAscendingIcon : slots.columnSortedDescendingIcon;
  return Icon ? /*#__PURE__*/(0, _jsxRuntime.jsx)(Icon, (0, _extends2.default)({}, other)) : null;
}
process.env.NODE_ENV !== "production" ? GridColumnUnsortedIcon.propTypes = {
  // ----------------------------- Warning --------------------------------
  // | These PropTypes are generated from the TypeScript type definitions |
  // | To update them edit the TypeScript types and run "pnpm proptypes"  |
  // ----------------------------------------------------------------------
  className: _propTypes.default.string,
  color: _propTypes.default.string,
  fontSize: _propTypes.default.oneOf(['inherit', 'large', 'medium', 'small']),
  sortingOrder: _propTypes.default.arrayOf(_propTypes.default.oneOf(['asc', 'desc'])).isRequired,
  style: _propTypes.default.object,
  titleAccess: _propTypes.default.string
} : void 0;