import _extends from "@babel/runtime/helpers/esm/extends";
import _objectWithoutPropertiesLoose from "@babel/runtime/helpers/esm/objectWithoutPropertiesLoose";
const _excluded = ["sortingOrder"];
import * as React from 'react';
import PropTypes from 'prop-types';
import { useGridRootProps } from "../hooks/utils/useGridRootProps.js";
import { jsx as _jsx } from "react/jsx-runtime";
function GridColumnUnsortedIcon(props) {
  const {
      sortingOrder
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded);
  const {
    slots
  } = useGridRootProps();
  const [nextSortDirection] = sortingOrder;
  const Icon = nextSortDirection === 'asc' ? slots.columnSortedAscendingIcon : slots.columnSortedDescendingIcon;
  return Icon ? /*#__PURE__*/_jsx(Icon, _extends({}, other)) : null;
}
process.env.NODE_ENV !== "production" ? GridColumnUnsortedIcon.propTypes = {
  // ----------------------------- Warning --------------------------------
  // | These PropTypes are generated from the TypeScript type definitions |
  // | To update them edit the TypeScript types and run "pnpm proptypes"  |
  // ----------------------------------------------------------------------
  className: PropTypes.string,
  color: PropTypes.string,
  fontSize: PropTypes.oneOf(['inherit', 'large', 'medium', 'small']),
  sortingOrder: PropTypes.arrayOf(PropTypes.oneOf(['asc', 'desc'])).isRequired,
  style: PropTypes.object,
  titleAccess: PropTypes.string
} : void 0;
export { GridColumnUnsortedIcon };