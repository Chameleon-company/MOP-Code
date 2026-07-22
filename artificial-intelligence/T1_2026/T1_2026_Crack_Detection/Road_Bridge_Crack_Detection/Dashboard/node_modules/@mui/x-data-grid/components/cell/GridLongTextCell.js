"use strict";
'use client';

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.GridLongTextCell = GridLongTextCell;
exports.renderLongTextCell = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var React = _interopRequireWildcard(require("react"));
var _clsx = _interopRequireDefault(require("clsx"));
var _composeClasses = _interopRequireDefault(require("@mui/utils/composeClasses"));
var _styles = require("@mui/material/styles");
var _gridClasses = require("../../constants/gridClasses");
var _useGridRootProps = require("../../hooks/utils/useGridRootProps");
var _useGridApiContext = require("../../hooks/utils/useGridApiContext");
var _useGridSelector = require("../../hooks/utils/useGridSelector");
var _gridDimensionsSelectors = require("../../hooks/features/dimensions/gridDimensionsSelectors");
var _assert = require("../../utils/assert");
var _cssVariables = require("../../constants/cssVariables");
var _jsxRuntime = require("react/jsx-runtime");
const useUtilityClasses = ownerState => {
  const {
    classes
  } = ownerState;
  const slots = {
    root: ['longTextCell'],
    content: ['longTextCellContent'],
    expandButton: ['longTextCellExpandButton'],
    collapseButton: ['longTextCellCollapseButton'],
    popup: ['longTextCellPopup'],
    popperContent: ['longTextCellPopperContent']
  };
  return (0, _composeClasses.default)(slots, _gridClasses.getDataGridUtilityClass, classes);
};
const GridLongTextCellRoot = (0, _styles.styled)('div', {
  name: 'MuiDataGrid',
  slot: 'LongTextCell'
})({
  display: 'flex',
  alignItems: 'center',
  width: '100%',
  height: '100%',
  position: 'relative'
});
const GridLongTextCellContent = (0, _styles.styled)('div', {
  name: 'MuiDataGrid',
  slot: 'LongTextCellContent'
})({
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
  flex: 1
});
const GridLongTextCellPopperContent = (0, _styles.styled)('div', {
  name: 'MuiDataGrid',
  slot: 'LongTextCellPopperContent'
})(({
  theme
}) => (0, _extends2.default)({}, theme.typography.body2, {
  letterSpacing: 'normal',
  paddingBlock: 15.5,
  paddingInline: 9,
  maxHeight: 52 * 3,
  overflow: 'auto',
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
  width: 'var(--_width)',
  border: `1px solid ${(theme.vars || theme).palette.divider}`,
  boxSizing: 'border-box'
}));
const GridLongTextCellCornerButton = (0, _styles.styled)('button', {
  name: 'MuiDataGrid',
  slot: 'LongTextCellCornerButton'
})(({
  theme
}) => ({
  lineHeight: 0,
  position: 'absolute',
  bottom: 1,
  right: 0,
  border: '1px solid',
  color: (theme.vars || theme).palette.text.secondary,
  borderColor: (theme.vars || theme).palette.divider,
  backgroundColor: (theme.vars || theme).palette.background.paper,
  borderRadius: 0,
  fontSize: '0.875rem',
  padding: 2,
  '&:focus-visible': {
    outline: 'none'
  },
  '&:hover': {
    backgroundColor: (theme.vars || theme).palette.background.paper,
    color: (theme.vars || theme).palette.text.primary
  },
  [`&.${_gridClasses.gridClasses.longTextCellExpandButton}`]: {
    right: -9,
    opacity: 0,
    [`.${_gridClasses.gridClasses.longTextCell}:hover &, .${_gridClasses.gridClasses.longTextCell}.Mui-focused &`]: {
      opacity: 1
    }
  },
  [`&.${_gridClasses.gridClasses.longTextCellCollapseButton}`]: {
    bottom: 2,
    right: 2,
    border: 'none'
  }
}));
const GridLongTextCellPopper = (0, _styles.styled)(_assert.NotRendered, {
  name: 'MuiDataGrid',
  slot: 'LongTextCellPopper'
})(({
  theme
}) => ({
  zIndex: _cssVariables.vars.zIndex.menu,
  background: (theme.vars || theme).palette.background.paper,
  '&[data-popper-reference-hidden]': {
    visibility: 'hidden',
    pointerEvents: 'none'
  }
}));
function GridLongTextCell(props) {
  const {
    id,
    value = '',
    colDef,
    hasFocus,
    slotProps,
    renderContent
  } = props;
  const popupId = `${id}-${colDef.field}-longtext-popup`;
  const rootProps = (0, _useGridRootProps.useGridRootProps)();
  const apiRef = (0, _useGridApiContext.useGridApiContext)();
  const classes = useUtilityClasses(rootProps);
  const rowHeight = (0, _useGridSelector.useGridSelector)(apiRef, _gridDimensionsSelectors.gridRowHeightSelector);
  const [popupOpen, setPopupOpen] = React.useState(false);
  const cellRef = React.useRef(null);
  const cornerButtonRef = React.useRef(null);
  React.useEffect(() => {
    if (hasFocus && !popupOpen) {
      if (cornerButtonRef.current && cornerButtonRef.current !== document.activeElement) {
        cornerButtonRef.current.focus();
      }
    }
    if (!hasFocus) {
      setPopupOpen(false);
    }
  }, [hasFocus, popupOpen]);
  const handleExpandClick = event => {
    // event.detail === 0 means keyboard-triggered click (Enter keyup on focused button)
    // Ignore these to prevent popup from opening when focus moves to this cell via Enter
    if (event.detail === 0) {
      return;
    }
    event.stopPropagation();
    setPopupOpen(true);
  };
  const handleExpandKeyDown = event => {
    if (event.key === ' ' && !event.shiftKey) {
      event.preventDefault(); // Prevent native button click on keyup
      event.stopPropagation(); // Prevent grid row selection
      setPopupOpen(prev => !prev);
    }
    if (event.key === 'Escape' && popupOpen) {
      event.stopPropagation(); // Prevent grid cell navigation
      setPopupOpen(false);
    }
  };
  const handleClickAway = () => {
    setPopupOpen(false);
  };
  const handleCollapseClick = event => {
    event.stopPropagation();
    setPopupOpen(false);
    apiRef.current.getCellElement(id, colDef.field)?.focus();
  };
  return /*#__PURE__*/(0, _jsxRuntime.jsxs)(GridLongTextCellRoot, (0, _extends2.default)({
    ref: cellRef
  }, slotProps?.root, {
    className: (0, _clsx.default)(classes.root, slotProps?.root?.className, hasFocus && 'Mui-focused'),
    children: [/*#__PURE__*/(0, _jsxRuntime.jsx)(GridLongTextCellContent, (0, _extends2.default)({}, slotProps?.content, {
      className: (0, _clsx.default)(classes.content, slotProps?.content?.className),
      children: value
    })), /*#__PURE__*/(0, _jsxRuntime.jsx)(GridLongTextCellCornerButton, (0, _extends2.default)({
      ref: cornerButtonRef,
      "aria-label": `${value}, ${apiRef.current.getLocaleText('longTextCellExpandLabel')}`,
      "aria-haspopup": "dialog",
      "aria-controls": popupOpen ? popupId : undefined,
      "aria-expanded": popupOpen,
      "aria-keyshortcuts": "Space",
      tabIndex: 0
    }, slotProps?.expandButton, {
      className: (0, _clsx.default)(classes.expandButton, slotProps?.expandButton?.className),
      onClick: handleExpandClick,
      onKeyDown: handleExpandKeyDown,
      children: /*#__PURE__*/(0, _jsxRuntime.jsx)(rootProps.slots.longTextCellExpandIcon, {
        fontSize: "inherit"
      })
    })), /*#__PURE__*/(0, _jsxRuntime.jsx)(GridLongTextCellPopper, (0, _extends2.default)({
      id: popupId,
      role: "dialog",
      "aria-label": colDef.headerName || colDef.field,
      as: rootProps.slots.basePopper,
      ownerState: rootProps,
      open: popupOpen,
      target: cellRef.current,
      placement: "bottom-start",
      onClickAway: handleClickAway,
      clickAwayMouseEvent: "onMouseDown",
      flip: true,
      material: {
        container: cellRef.current?.closest('[role="row"]'),
        modifiers: [{
          name: 'offset',
          options: {
            offset: [-10, -rowHeight]
          }
        }]
      }
    }, slotProps?.popper, {
      className: (0, _clsx.default)(classes.popup, slotProps?.popper?.className),
      children: /*#__PURE__*/(0, _jsxRuntime.jsxs)(GridLongTextCellPopperContent, (0, _extends2.default)({
        tabIndex: -1,
        onKeyDown: event => {
          if (event.key === 'Escape') {
            event.stopPropagation();
            setPopupOpen(false);
            apiRef.current.getCellElement(id, colDef.field)?.focus();
          }
        }
      }, slotProps?.popperContent, {
        className: (0, _clsx.default)(classes.popperContent, slotProps?.popperContent?.className),
        style: (0, _extends2.default)({
          '--_width': `${colDef.computedWidth}px`
        }, slotProps?.popperContent?.style),
        children: [renderContent ? renderContent(value) : value, /*#__PURE__*/(0, _jsxRuntime.jsx)(GridLongTextCellCornerButton, (0, _extends2.default)({
          "aria-label": apiRef.current.getLocaleText('longTextCellCollapseLabel'),
          "aria-keyshortcuts": "Escape"
        }, slotProps?.collapseButton, {
          className: (0, _clsx.default)(classes.collapseButton, slotProps?.collapseButton?.className),
          onClick: handleCollapseClick,
          children: /*#__PURE__*/(0, _jsxRuntime.jsx)(rootProps.slots.longTextCellCollapseIcon, {
            fontSize: "inherit"
          })
        }))]
      }))
    }))]
  }));
}
const renderLongTextCell = params => /*#__PURE__*/(0, _jsxRuntime.jsx)(GridLongTextCell, (0, _extends2.default)({}, params));
exports.renderLongTextCell = renderLongTextCell;
if (process.env.NODE_ENV !== "production") renderLongTextCell.displayName = "renderLongTextCell";