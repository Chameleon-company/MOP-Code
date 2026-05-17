'use client';

import _extends from "@babel/runtime/helpers/esm/extends";
import _objectWithoutPropertiesLoose from "@babel/runtime/helpers/esm/objectWithoutPropertiesLoose";
const _excluded = ["id", "value", "formattedValue", "api", "field", "row", "rowNode", "colDef", "cellMode", "isEditable", "tabIndex", "hasFocus", "isValidating", "debounceMs", "isProcessingProps", "onValueChange", "slotProps"];
import * as React from 'react';
import composeClasses from '@mui/utils/composeClasses';
import useEnhancedEffect from '@mui/utils/useEnhancedEffect';
import { styled } from '@mui/material/styles';
import { forwardRef } from '@mui/x-internals/forwardRef';
import { NotRendered } from "../../utils/assert.js";
import { vars } from "../../constants/cssVariables.js";
import { getDataGridUtilityClass } from "../../constants/gridClasses.js";
import { useGridRootProps } from "../../hooks/utils/useGridRootProps.js";
import { useGridApiContext } from "../../hooks/utils/useGridApiContext.js";
import { jsx as _jsx } from "react/jsx-runtime";
const useUtilityClasses = ownerState => {
  const {
    classes
  } = ownerState;
  const slots = {
    root: ['editInputCell']
  };
  return composeClasses(slots, getDataGridUtilityClass, classes);
};
const GridEditInputCellRoot = styled(NotRendered, {
  name: 'MuiDataGrid',
  slot: 'EditInputCell'
})({
  font: vars.typography.font.body,
  padding: '1px 0',
  '& input': {
    padding: '0 16px',
    height: '100%'
  }
});
const GridEditInputCell = forwardRef((props, ref) => {
  const rootProps = useGridRootProps();
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
    other = _objectWithoutPropertiesLoose(props, _excluded);
  const apiRef = useGridApiContext();
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
  useEnhancedEffect(() => {
    if (hasFocus) {
      inputRef.current.focus();
    }
  }, [hasFocus]);
  return /*#__PURE__*/_jsx(GridEditInputCellRoot, _extends({
    as: rootProps.slots.baseInput,
    inputRef: inputRef,
    className: classes.root,
    ownerState: rootProps,
    fullWidth: true,
    type: colDef.type === 'number' ? colDef.type : 'text',
    value: valueState ?? '',
    onChange: handleChange,
    endAdornment: isProcessingProps ? /*#__PURE__*/_jsx(rootProps.slots.loadIcon, {
      fontSize: "small",
      color: "action"
    }) : undefined
  }, other, slotProps?.root, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") GridEditInputCell.displayName = "GridEditInputCell";
export { GridEditInputCell };
export const renderEditInputCell = params => /*#__PURE__*/_jsx(GridEditInputCell, _extends({}, params));
if (process.env.NODE_ENV !== "production") renderEditInputCell.displayName = "renderEditInputCell";