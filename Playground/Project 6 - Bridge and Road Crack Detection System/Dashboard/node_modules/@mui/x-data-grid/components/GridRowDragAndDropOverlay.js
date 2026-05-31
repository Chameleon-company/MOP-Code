"use strict";
'use client';

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.GridRowDragAndDropOverlay = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var React = _interopRequireWildcard(require("react"));
var _styles = require("@mui/material/styles");
var _createStyled = require("@mui/system/createStyled");
var _useGridPrivateApiContext = require("../hooks/utils/useGridPrivateApiContext");
var _useGridSelector = require("../hooks/utils/useGridSelector");
var _gridRowReorderSelector = require("../hooks/features/rowReorder/gridRowReorderSelector");
var _jsxRuntime = require("react/jsx-runtime");
const GridRowDragAndDropOverlayRoot = (0, _styles.styled)('div', {
  name: 'MuiDataGrid',
  slot: 'RowDragOverlay',
  shouldForwardProp: prop => (0, _createStyled.shouldForwardProp)(prop) && prop !== 'action'
})(({
  theme,
  action
}) => (0, _extends2.default)({
  position: 'absolute',
  left: 0,
  right: 0,
  top: 0,
  bottom: 0,
  pointerEvents: 'none',
  zIndex: 1
}, action === 'above' && {
  '&::before': {
    pointerEvents: 'none',
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: '2px',
    backgroundColor: (theme.vars || theme).palette.primary.main
  }
}, action === 'below' && {
  '&::after': {
    pointerEvents: 'none',
    content: '""',
    position: 'absolute',
    bottom: '-2px',
    left: 0,
    right: 0,
    height: '2px',
    backgroundColor: (theme.vars || theme).palette.primary.main
  }
}, action === 'inside' && {
  backgroundColor: theme.vars ? `rgba(${theme.vars.palette.primary.mainChannel} / 0.1)` : (0, _styles.alpha)(theme.palette.primary.main, 0.1)
}));
const GridRowDragAndDropOverlay = exports.GridRowDragAndDropOverlay = /*#__PURE__*/React.memo(function GridRowDragAndDropOverlay(props) {
  const {
    rowId,
    className
  } = props;
  const apiRef = (0, _useGridPrivateApiContext.useGridPrivateApiContext)();
  const dropPosition = (0, _useGridSelector.useGridSelector)(apiRef, _gridRowReorderSelector.gridRowDropPositionSelector, rowId);
  if (!dropPosition) {
    return null;
  }
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(GridRowDragAndDropOverlayRoot, {
    action: dropPosition,
    className: className
  });
});
if (process.env.NODE_ENV !== "production") GridRowDragAndDropOverlay.displayName = "GridRowDragAndDropOverlay";