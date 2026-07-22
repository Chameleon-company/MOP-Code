"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.getColumnMenuItemKeys = getColumnMenuItemKeys;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
/**
 * Returns the list of column menu item keys (sorted by `displayOrder`) that should be rendered for a given column.
 * This is shared between the column header (to know if menu is empty) and the menu itself (to render items).
 */
function getColumnMenuItemKeys(params) {
  const {
    apiRef,
    colDef,
    defaultSlots,
    defaultSlotProps,
    slots = {},
    slotProps = {}
  } = params;
  const processedComponents = (0, _extends2.default)({}, defaultSlots, slots);
  let processedSlotProps = defaultSlotProps;
  if (slotProps && Object.keys(slotProps).length > 0) {
    const mergedProps = (0, _extends2.default)({}, slotProps);
    Object.entries(defaultSlotProps).forEach(([key, currentSlotProps]) => {
      mergedProps[key] = (0, _extends2.default)({}, currentSlotProps, slotProps[key] || {});
    });
    processedSlotProps = mergedProps;
  }
  const defaultItems = apiRef.current.unstable_applyPipeProcessors('columnMenu', [], colDef);
  const defaultComponentKeys = Object.keys(defaultSlots);
  const userItems = Object.keys(slots).filter(key => !defaultComponentKeys.includes(key));
  const uniqueItems = Array.from(new Set([...defaultItems, ...userItems]));
  const cleansedItems = uniqueItems.filter(key => processedComponents[key] != null);
  return cleansedItems.sort((a, b) => {
    const leftItemProps = processedSlotProps[a];
    const rightItemProps = processedSlotProps[b];
    const leftDisplayOrder = Number.isFinite(leftItemProps?.displayOrder) ? leftItemProps.displayOrder : 100;
    const rightDisplayOrder = Number.isFinite(rightItemProps?.displayOrder) ? rightItemProps.displayOrder : 100;
    return leftDisplayOrder - rightDisplayOrder;
  });
}