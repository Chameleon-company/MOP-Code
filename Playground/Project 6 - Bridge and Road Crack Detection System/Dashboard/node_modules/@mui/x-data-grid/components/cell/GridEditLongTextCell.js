"use strict";
'use client';

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.GridEditLongTextCell = GridEditLongTextCell;
exports.renderEditLongTextCell = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var React = _interopRequireWildcard(require("react"));
var _clsx = _interopRequireDefault(require("clsx"));
var _composeClasses = _interopRequireDefault(require("@mui/utils/composeClasses"));
var _useEnhancedEffect = _interopRequireDefault(require("@mui/utils/useEnhancedEffect"));
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
    root: ['editLongTextCell'],
    value: ['editLongTextCellValue'],
    popup: ['editLongTextCellPopup'],
    popperContent: ['editLongTextCellPopperContent'],
    textarea: ['editLongTextCellTextarea']
  };
  return (0, _composeClasses.default)(slots, _gridClasses.getDataGridUtilityClass, classes);
};
const GridEditLongTextCellTextarea = (0, _styles.styled)(_assert.NotRendered, {
  name: 'MuiDataGrid',
  slot: 'EditLongTextCellTextarea'
})(({
  theme
}) => (0, _extends2.default)({
  width: '100%',
  padding: 0
}, theme.typography.body2, {
  letterSpacing: 'normal',
  outline: 'none',
  background: 'transparent',
  border: 'none',
  resize: 'vertical'
}));
const GridEditLongTextCellRoot = (0, _styles.styled)('div', {
  name: 'MuiDataGrid',
  slot: 'EditLongTextCell'
})({
  display: 'flex',
  alignItems: 'center',
  width: '100%',
  height: '100%',
  position: 'relative'
});
const GridEditLongTextCellValue = (0, _styles.styled)('div', {
  name: 'MuiDataGrid',
  slot: 'EditLongTextCellValue'
})({
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
  width: '100%',
  paddingInline: 10
});
const GridEditLongTextCellPopper = (0, _styles.styled)(_assert.NotRendered, {
  name: 'MuiDataGrid',
  slot: 'EditLongTextCellPopper'
})(({
  theme
}) => ({
  zIndex: _cssVariables.vars.zIndex.menu,
  background: (theme.vars || theme).palette.background.paper,
  '&[data-popper-reference-hidden]': {
    opacity: 0 // use opacity to preserve focus.
  }
}));
const GridEditLongTextCellPopperContent = (0, _styles.styled)('div', {
  name: 'MuiDataGrid',
  slot: 'EditLongTextCellPopperContent'
})(({
  theme
}) => (0, _extends2.default)({}, theme.typography.body2, {
  letterSpacing: 'normal',
  paddingBlock: 15.5,
  paddingInline: 9,
  height: 'max-content',
  overflow: 'auto',
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
  width: 'var(--_width)',
  border: `1px solid ${(theme.vars || theme).palette.divider}`,
  boxShadow: (theme.vars || theme).shadows[4],
  boxSizing: 'border-box'
}));
function GridEditLongTextCell(props) {
  const {
    id,
    value,
    field,
    colDef,
    hasFocus,
    cellMode,
    slotProps
  } = props;
  const rootProps = (0, _useGridRootProps.useGridRootProps)();
  const apiRef = (0, _useGridApiContext.useGridApiContext)();
  const classes = useUtilityClasses(rootProps);
  const rowHeight = (0, _useGridSelector.useGridSelector)(apiRef, _gridDimensionsSelectors.gridRowHeightSelector);
  const [valueState, setValueState] = React.useState(value);
  const [anchorEl, setAnchorEl] = React.useState(null);
  const meta = apiRef.current.unstable_getEditCellMeta(id, field);
  const popupId = `${id}-${field}-longtext-edit-popup`;

  // Only show popup when this cell has focus
  // This fixes editMode="row" where all cells enter edit mode simultaneously
  const showPopup = hasFocus && Boolean(anchorEl);
  React.useEffect(() => {
    if (meta?.changeReason !== 'debouncedSetEditCellValue') {
      setValueState(value);
    }
  }, [meta, value]);
  return /*#__PURE__*/(0, _jsxRuntime.jsxs)(GridEditLongTextCellRoot, (0, _extends2.default)({
    tabIndex: cellMode === 'edit' && rootProps.editMode === 'row' ? 0 : undefined,
    ref: setAnchorEl,
    "aria-controls": showPopup ? popupId : undefined,
    "aria-expanded": showPopup
  }, slotProps?.root, {
    className: (0, _clsx.default)(classes.root, slotProps?.root?.className),
    children: [/*#__PURE__*/(0, _jsxRuntime.jsx)(GridEditLongTextCellValue, (0, _extends2.default)({}, slotProps?.value, {
      className: (0, _clsx.default)(classes.value, slotProps?.value?.className),
      children: valueState
    })), /*#__PURE__*/(0, _jsxRuntime.jsx)(GridEditLongTextCellPopper, (0, _extends2.default)({
      as: rootProps.slots.basePopper,
      ownerState: rootProps,
      id: popupId,
      role: "dialog",
      "aria-label": colDef.headerName || field,
      open: showPopup,
      target: anchorEl,
      placement: "bottom-start",
      flip: true,
      material: {
        container: anchorEl?.closest('[role="row"]'),
        modifiers: [{
          name: 'offset',
          options: {
            offset: [-1, -rowHeight]
          }
        }]
      }
    }, slotProps?.popper, {
      className: (0, _clsx.default)(classes.popup, slotProps?.popper?.className),
      children: /*#__PURE__*/(0, _jsxRuntime.jsx)(GridEditLongTextCellPopperContent, (0, _extends2.default)({}, slotProps?.popperContent, {
        className: (0, _clsx.default)(classes.popperContent, slotProps?.popperContent?.className),
        style: {
          '--_width': `${colDef.computedWidth}px`
        },
        children: /*#__PURE__*/(0, _jsxRuntime.jsx)(GridEditLongTextarea, (0, _extends2.default)({}, props, {
          valueState: valueState,
          setValueState: setValueState
        }))
      }))
    }))]
  }));
}
function GridEditLongTextarea(props) {
  const {
    id,
    field,
    colDef,
    debounceMs = 200,
    onValueChange,
    valueState,
    setValueState,
    hasFocus,
    slotProps
  } = props;
  const textareaRef = React.useRef(null);
  const apiRef = (0, _useGridApiContext.useGridApiContext)();
  const rootProps = (0, _useGridRootProps.useGridRootProps)();
  const classes = useUtilityClasses(rootProps);
  (0, _useEnhancedEffect.default)(() => {
    if (hasFocus && textareaRef.current) {
      // preventScroll: the popper is portaled into the GridRow, so focusing
      // without it triggers the browser to scroll the grid container which is undesirable.
      textareaRef.current.focus({
        preventScroll: true
      });
      // Move cursor to end of text
      const length = textareaRef.current.value.length;
      textareaRef.current.setSelectionRange(length, length);
    }
  }, [hasFocus]);
  const handleChange = React.useCallback(async event => {
    const newValue = event.target.value;
    const column = apiRef.current.getColumn(field);
    let parsedValue = newValue;
    if (column.valueParser) {
      parsedValue = column.valueParser(newValue, apiRef.current.getRow(id), column, apiRef);
    }
    setValueState(parsedValue);
    apiRef.current.setEditCellValue({
      id,
      field,
      value: parsedValue,
      debounceMs,
      unstable_skipValueParser: true
    }, event);
    if (onValueChange) {
      await onValueChange(event, newValue);
    }
  }, [apiRef, debounceMs, field, id, onValueChange, setValueState]);
  const handleKeyDown = React.useCallback(event => {
    if (event.key === 'Enter' && event.shiftKey) {
      // Shift+Enter: let textarea handle newline, stop propagation to prevent grid from exiting edit
      event.stopPropagation();
    }
    if (rootProps.editMode === 'cell' && event.key === 'Escape') {
      apiRef.current.stopCellEditMode({
        id,
        field,
        ignoreModifications: true
      });
    }
  }, [apiRef, field, id, rootProps.editMode]);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(GridEditLongTextCellTextarea, (0, _extends2.default)({
    ref: textareaRef,
    as: rootProps.slots.baseTextarea,
    ownerState: rootProps,
    "aria-label": colDef.headerName || field,
    value: valueState ?? '',
    onChange: handleChange,
    onKeyDown: handleKeyDown
  }, slotProps?.textarea, {
    className: (0, _clsx.default)(classes.textarea, slotProps?.textarea?.className)
  }));
}
const renderEditLongTextCell = params => /*#__PURE__*/(0, _jsxRuntime.jsx)(GridEditLongTextCell, (0, _extends2.default)({}, params));
exports.renderEditLongTextCell = renderEditLongTextCell;
if (process.env.NODE_ENV !== "production") renderEditLongTextCell.displayName = "renderEditLongTextCell";