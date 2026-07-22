'use client';

import _extends from "@babel/runtime/helpers/esm/extends";
import _objectWithoutPropertiesLoose from "@babel/runtime/helpers/esm/objectWithoutPropertiesLoose";
const _excluded = ["formattedValue", "colDef", "cellMode", "row", "api", "id", "value", "rowNode", "field", "hasFocus", "tabIndex", "isEditable"];
import * as React from 'react';
import composeClasses from '@mui/utils/composeClasses';
import { styled } from '@mui/material/styles';
import { vars } from "../../constants/cssVariables.js";
import { useGridRootProps } from "../../hooks/utils/useGridRootProps.js";
import { getDataGridUtilityClass } from "../../constants/gridClasses.js";
import { jsx as _jsx } from "react/jsx-runtime";
const GridFooterCellRoot = styled('div', {
  name: 'MuiDataGrid',
  slot: 'FooterCell'
})({
  fontWeight: vars.typography.fontWeight.medium,
  color: vars.colors.foreground.accent
});
const useUtilityClasses = ownerState => {
  const {
    classes
  } = ownerState;
  const slots = {
    root: ['footerCell']
  };
  return composeClasses(slots, getDataGridUtilityClass, classes);
};
function GridFooterCellRaw(props) {
  const {
      formattedValue
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded);
  const rootProps = useGridRootProps();
  const ownerState = {
    classes: rootProps.classes
  };
  const classes = useUtilityClasses(ownerState);
  return /*#__PURE__*/_jsx(GridFooterCellRoot, _extends({
    ownerState: ownerState,
    className: classes.root
  }, other, {
    children: formattedValue
  }));
}
const GridFooterCell = /*#__PURE__*/React.memo(GridFooterCellRaw);
if (process.env.NODE_ENV !== "production") GridFooterCell.displayName = "GridFooterCell";
export { GridFooterCell };