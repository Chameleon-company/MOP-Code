"use strict";
'use client';

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.GridCellCheckboxRenderer = exports.GridCellCheckboxForwardRef = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var _objectWithoutPropertiesLoose2 = _interopRequireDefault(require("@babel/runtime/helpers/objectWithoutPropertiesLoose"));
var React = _interopRequireWildcard(require("react"));
var _propTypes = _interopRequireDefault(require("prop-types"));
var _clsx = _interopRequireDefault(require("clsx"));
var _composeClasses = _interopRequireDefault(require("@mui/utils/composeClasses"));
var _useEventCallback = _interopRequireDefault(require("@mui/utils/useEventCallback"));
var _forwardRef = require("@mui/x-internals/forwardRef");
var _useGridApiContext = require("../../hooks/utils/useGridApiContext");
var _useGridRootProps = require("../../hooks/utils/useGridRootProps");
var _gridClasses = require("../../constants/gridClasses");
var _useGridSelector = require("../../hooks/utils/useGridSelector");
var _utils = require("../../hooks/features/rowSelection/utils");
var _jsxRuntime = require("react/jsx-runtime");
const _excluded = ["field", "id", "formattedValue", "row", "rowNode", "colDef", "isEditable", "cellMode", "hasFocus", "tabIndex", "api"];
const useUtilityClasses = ownerState => {
  const {
    classes
  } = ownerState;
  const slots = {
    root: ['checkboxInput']
  };
  return (0, _composeClasses.default)(slots, _gridClasses.getDataGridUtilityClass, classes);
};
const GridCellCheckboxForwardRef = exports.GridCellCheckboxForwardRef = (0, _forwardRef.forwardRef)(function GridCellCheckboxRenderer(props, ref) {
  const {
      field,
      id,
      rowNode,
      tabIndex
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded);
  const apiRef = (0, _useGridApiContext.useGridApiContext)();
  const rootProps = (0, _useGridRootProps.useGridRootProps)();
  const ownerState = {
    classes: rootProps.classes
  };
  const classes = useUtilityClasses(ownerState);
  const {
    isIndeterminate,
    isChecked,
    isSelectable
  } = (0, _useGridSelector.useGridSelector)(apiRef, _utils.checkboxPropsSelector, {
    groupId: id,
    autoSelectParents: rootProps.rowSelectionPropagation?.parents ?? false
  }, _useGridSelector.objectShallowCompare);
  const disabled = !isSelectable;
  const handleChange = event => {
    if (disabled) {
      return;
    }
    const params = {
      value: event.target.checked,
      id
    };
    apiRef.current.publishEvent('rowSelectionCheckboxChange', params, event);
  };
  React.useLayoutEffect(() => {
    if (tabIndex === 0 && !disabled) {
      const element = apiRef.current.getCellElement(id, field);
      if (element) {
        element.tabIndex = -1;
      }
    }
  }, [apiRef, tabIndex, id, field, disabled]);
  const handleKeyDown = (0, _useEventCallback.default)(event => {
    if (event.key === ' ') {
      // We call event.stopPropagation to avoid selecting the row and also scrolling to bottom
      // TODO: Remove and add a check inside useGridKeyboardNavigation
      event.stopPropagation();
    }
    if (disabled) {
      return;
    }
  });
  const handleClick = (0, _useEventCallback.default)(event => {
    if (disabled) {
      event.preventDefault();
      return;
    }
  });
  const handleMouseDown = (0, _useEventCallback.default)(() => {
    if (disabled) {
      return;
    }
  });
  if (rowNode.type === 'footer' || rowNode.type === 'pinnedRow') {
    return null;
  }
  const label = apiRef.current.getLocaleText(isChecked && !isIndeterminate ? 'checkboxSelectionUnselectRow' : 'checkboxSelectionSelectRow');
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(rootProps.slots.baseCheckbox, (0, _extends2.default)({
    tabIndex: disabled ? -1 : tabIndex,
    checked: isChecked && !isIndeterminate,
    onChange: handleChange,
    onClick: handleClick,
    onMouseDown: handleMouseDown,
    className: (0, _clsx.default)(classes.root, disabled && 'Mui-disabled'),
    disabled: disabled,
    material: {
      disableRipple: disabled
    },
    slotProps: {
      htmlInput: {
        'aria-disabled': disabled || undefined,
        'aria-label': label,
        name: 'select_row'
      }
    },
    onKeyDown: handleKeyDown,
    indeterminate: isIndeterminate
  }, rootProps.slotProps?.baseCheckbox, other, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") GridCellCheckboxForwardRef.displayName = "GridCellCheckboxForwardRef";
process.env.NODE_ENV !== "production" ? GridCellCheckboxForwardRef.propTypes = {
  // ----------------------------- Warning --------------------------------
  // | These PropTypes are generated from the TypeScript type definitions |
  // | To update them edit the TypeScript types and run "pnpm proptypes"  |
  // ----------------------------------------------------------------------
  /**
   * GridApi that let you manipulate the grid.
   */
  api: _propTypes.default.object.isRequired,
  /**
   * The mode of the cell.
   */
  cellMode: _propTypes.default.oneOf(['edit', 'view']).isRequired,
  /**
   * The column of the row that the current cell belongs to.
   */
  colDef: _propTypes.default.object.isRequired,
  /**
   * The column field of the cell that triggered the event.
   */
  field: _propTypes.default.string.isRequired,
  /**
   * The cell value formatted with the column valueFormatter.
   */
  formattedValue: _propTypes.default.any,
  /**
   * If true, the cell is the active element.
   */
  hasFocus: _propTypes.default.bool.isRequired,
  /**
   * The grid row id.
   */
  id: _propTypes.default.oneOfType([_propTypes.default.number, _propTypes.default.string]).isRequired,
  /**
   * If true, the cell is editable.
   */
  isEditable: _propTypes.default.bool,
  /**
   * The row model of the row that the current cell belongs to.
   */
  row: _propTypes.default.any.isRequired,
  /**
   * The node of the row that the current cell belongs to.
   */
  rowNode: _propTypes.default.object.isRequired,
  /**
   * the tabIndex value.
   */
  tabIndex: _propTypes.default.oneOf([-1, 0]).isRequired,
  /**
   * The cell value.
   * If the column has `valueGetter`, use `params.row` to directly access the fields.
   */
  value: _propTypes.default.any
} : void 0;
const GridCellCheckboxRenderer = exports.GridCellCheckboxRenderer = GridCellCheckboxForwardRef;