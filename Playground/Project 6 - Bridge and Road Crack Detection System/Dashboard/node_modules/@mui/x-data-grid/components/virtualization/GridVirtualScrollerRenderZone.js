"use strict";
'use client';

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.GridVirtualScrollerRenderZone = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var React = _interopRequireWildcard(require("react"));
var _clsx = _interopRequireDefault(require("clsx"));
var _system = require("@mui/system");
var _composeClasses = _interopRequireDefault(require("@mui/utils/composeClasses"));
var _forwardRef = require("@mui/x-internals/forwardRef");
var _useGridRootProps = require("../../hooks/utils/useGridRootProps");
var _gridClasses = require("../../constants/gridClasses");
var _jsxRuntime = require("react/jsx-runtime");
const useUtilityClasses = ownerState => {
  const {
    classes
  } = ownerState;
  const slots = {
    root: ['virtualScrollerRenderZone']
  };
  return (0, _composeClasses.default)(slots, _gridClasses.getDataGridUtilityClass, classes);
};
const VirtualScrollerRenderZoneRoot = (0, _system.styled)('div', {
  name: 'MuiDataGrid',
  slot: 'VirtualScrollerRenderZone'
})({
  position: 'absolute',
  display: 'flex',
  // Prevents margin collapsing when using `getRowSpacing`
  flexDirection: 'column'
});
const GridVirtualScrollerRenderZone = exports.GridVirtualScrollerRenderZone = (0, _forwardRef.forwardRef)(function GridVirtualScrollerRenderZone(props, ref) {
  const rootProps = (0, _useGridRootProps.useGridRootProps)();
  const classes = useUtilityClasses(rootProps);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(VirtualScrollerRenderZoneRoot, (0, _extends2.default)({
    ownerState: rootProps
  }, props, {
    className: (0, _clsx.default)(classes.root, props.className),
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") GridVirtualScrollerRenderZone.displayName = "GridVirtualScrollerRenderZone";