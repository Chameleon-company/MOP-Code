"use strict";
'use client';

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.renderEditInputCell = exports.GridEditInputCell = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var _objectWithoutPropertiesLoose2 = _interopRequireDefault(require("@babel/runtime/helpers/objectWithoutPropertiesLoose"));
var React = _interopRequireWildcard(require("react"));
var _composeClasses = _interopRequireDefault(require("@mui/utils/composeClasses"));
var _useEnhancedEffect = _interopRequireDefault(require("@mui/utils/useEnhancedEffect"));
var _styles = require("@mui/material/styles");
var _forwardRef = require("@mui/x-internals/forwardRef");
var _assert = require("../../utils/assert");
var _cssVariables = require("../../constants/cssVariables");
var _gridClasses = require("../../constants/gridClasses");
var _useGridRootProps = require("../../hooks/utils/useGridRootProps");
var _useGridApiContext = require("../../hooks/utils/useGridApiContext");
var _jsxRuntime = require("react/jsx-runtime");
const _excluded = ["id", "value", "formattedValue", "api", "field", "row", "rowNode", "colDef", "cellMode", "isEditable", "tabIndex", "hasFocus", "isValidating", "debounceMs", "isProcessingProps", "onValueChange", "slotProps"];
const useUtilityClasses = ownerState => {
  const {
    classes
  } = ownerState;
  const slots = {
    root: ['editInputCell']
  };
  return (0, _composeClasses.default)(slots, _gridClasses.getDataGridUtilityClass, classes);
};
const GridEditInputCellRoot = (0, _styles.styled)(_assert.NotRendered, {
  name: 'MuiDataGrid',
  slot: 'EditInputCell'
})({
  font: _cssVariables.vars.typography.font.body,
  padding: '1px 0',
  '& input': {
    padding: '0 16px',
    height: '100%'
  }
});
const GridEditInputCell = exports.GridEditInputCell = (0, _forwardRef.forwardRef)((props, ref) => {
  const rootProps = (0, _useGridRootProps.useGridRootProps)();
  const {
      id,
      value,
      field,
      colDef,
      hasFocus,
      debounceMs = 200,
      isProcessingProps,
      onValueChange,
      slotProps
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded);
  const apiRef = (0, _useGridApiContext.useGridApiContext)();
  const inputRef = React.useRef(null);
  const [valueState, setValueState] = React.useState(value);
  const classes = useUtilityClasses(rootProps);
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
  }, [apiRef, debounceMs, field, id, onValueChange]);
  const meta = apiRef.current.unstable_getEditCellMeta(id, field);
  React.useEffect(() => {
    if (meta?.changeReason !== 'debouncedSetEditCellValue') {
      setValueState(value);
    }
  }, [meta, value]);
  (0, _useEnhancedEffect.default)(() => {
    if (hasFocus) {
      inputRef.current.focus();
    }
  }, [hasFocus]);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(GridEditInputCellRoot, (0, _extends2.default)({
    as: rootProps.slots.baseInput,
    inputRef: inputRef,
    className: classes.root,
    ownerState: rootProps,
    fullWidth: true,
    type: colDef.type === 'number' ? colDef.type : 'text',
    value: valueState ?? '',
    onChange: handleChange,
    endAdornment: isProcessingProps ? /*#__PURE__*/(0, _jsxRuntime.jsx)(rootProps.slots.loadIcon, {
      fontSize: "small",
      color: "action"
    }) : undefined
  }, other, slotProps?.root, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") GridEditInputCell.displayName = "GridEditInputCell";
const renderEditInputCell = params => /*#__PURE__*/(0, _jsxRuntime.jsx)(GridEditInputCell, (0, _extends2.default)({}, params));
exports.renderEditInputCell = renderEditInputCell;
if (process.env.NODE_ENV !== "production") renderEditInputCell.displayName = "renderEditInputCell";