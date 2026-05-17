'use client';

import _extends from "@babel/runtime/helpers/esm/extends";
import * as React from 'react';
import clsx from 'clsx';
import { styled } from '@mui/system';
import composeClasses from '@mui/utils/composeClasses';
import { forwardRef } from '@mui/x-internals/forwardRef';
import { useGridRootProps } from "../../hooks/utils/useGridRootProps.js";
import { getDataGridUtilityClass } from "../../constants/gridClasses.js";
import { jsx as _jsx } from "react/jsx-runtime";
const useUtilityClasses = ownerState => {
  const {
    classes
  } = ownerState;
  const slots = {
    root: ['virtualScrollerRenderZone']
  };
  return composeClasses(slots, getDataGridUtilityClass, classes);
};
const VirtualScrollerRenderZoneRoot = styled('div', {
  name: 'MuiDataGrid',
  slot: 'VirtualScrollerRenderZone'
})({
  position: 'absolute',
  display: 'flex',
  // Prevents margin collapsing when using `getRowSpacing`
  flexDirection: 'column'
});
const GridVirtualScrollerRenderZone = forwardRef(function GridVirtualScrollerRenderZone(props, ref) {
  const rootProps = useGridRootProps();
  const classes = useUtilityClasses(rootProps);
  return /*#__PURE__*/_jsx(VirtualScrollerRenderZoneRoot, _extends({
    ownerState: rootProps
  }, props, {
    className: clsx(classes.root, props.className),
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") GridVirtualScrollerRenderZone.displayName = "GridVirtualScrollerRenderZone";
export { GridVirtualScrollerRenderZone };