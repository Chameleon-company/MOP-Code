"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.GridScrollbarFillerCell = GridScrollbarFillerCell;
var _clsx = _interopRequireDefault(require("clsx"));
var _constants = require("../constants");
var _jsxRuntime = require("react/jsx-runtime");
const classes = {
  root: _constants.gridClasses.scrollbarFiller,
  pinnedRight: _constants.gridClasses['scrollbarFiller--pinnedRight']
};
function GridScrollbarFillerCell({
  pinnedRight
}) {
  return /*#__PURE__*/(0, _jsxRuntime.jsx)("div", {
    role: "presentation",
    className: (0, _clsx.default)(classes.root, pinnedRight && classes.pinnedRight)
  });
}