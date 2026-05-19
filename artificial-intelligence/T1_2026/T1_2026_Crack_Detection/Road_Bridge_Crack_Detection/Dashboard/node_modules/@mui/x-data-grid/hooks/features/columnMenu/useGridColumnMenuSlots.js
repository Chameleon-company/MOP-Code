"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useGridColumnMenuSlots = void 0;
var _objectWithoutPropertiesLoose2 = _interopRequireDefault(require("@babel/runtime/helpers/objectWithoutPropertiesLoose"));
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var React = _interopRequireWildcard(require("react"));
var _useGridRootProps = require("../../utils/useGridRootProps");
var _useGridPrivateApiContext = require("../../utils/useGridPrivateApiContext");
var _getColumnMenuItemKeys = require("./getColumnMenuItemKeys");
const _excluded = ["displayOrder"];
const useGridColumnMenuSlots = props => {
  const apiRef = (0, _useGridPrivateApiContext.useGridPrivateApiContext)();
  const rootProps = (0, _useGridRootProps.useGridRootProps)();
  const {
    defaultSlots,
    defaultSlotProps,
    slots = {},
    slotProps = {},
    hideMenu,
    colDef,
    addDividers = true
  } = props;
  const processedComponents = React.useMemo(() => (0, _extends2.default)({}, defaultSlots, slots), [defaultSlots, slots]);
  const processedSlotProps = React.useMemo(() => {
    if (!slotProps || Object.keys(slotProps).length === 0) {
      return defaultSlotProps;
    }
    const mergedProps = (0, _extends2.default)({}, slotProps);
    Object.entries(defaultSlotProps).forEach(([key, currentSlotProps]) => {
      mergedProps[key] = (0, _extends2.default)({}, currentSlotProps, slotProps[key] || {});
    });
    return mergedProps;
  }, [defaultSlotProps, slotProps]);
  return React.useMemo(() => {
    const sortedKeys = (0, _getColumnMenuItemKeys.getColumnMenuItemKeys)({
      apiRef,
      colDef,
      defaultSlots,
      defaultSlotProps,
      slots,
      slotProps
    });
    return sortedKeys.reduce((acc, key, index) => {
      let itemProps = {
        colDef,
        onClick: hideMenu
      };
      const processedComponentProps = processedSlotProps[key];
      if (processedComponentProps) {
        const customProps = (0, _objectWithoutPropertiesLoose2.default)(processedComponentProps, _excluded);
        itemProps = (0, _extends2.default)({}, itemProps, customProps);
      }
      return addDividers && index !== sortedKeys.length - 1 ? [...acc, [processedComponents[key], itemProps], [rootProps.slots.baseDivider, {}]] : [...acc, [processedComponents[key], itemProps]];
    }, []);
  }, [addDividers, apiRef, colDef, defaultSlotProps, defaultSlots, hideMenu, processedComponents, processedSlotProps, slotProps, slots, rootProps.slots.baseDivider]);
};
exports.useGridColumnMenuSlots = useGridColumnMenuSlots;