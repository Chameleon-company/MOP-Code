"use strict";
'use client';

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.GridActionsCell = GridActionsCell;
exports.renderActionsCell = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var _objectWithoutPropertiesLoose2 = _interopRequireDefault(require("@babel/runtime/helpers/objectWithoutPropertiesLoose"));
var React = _interopRequireWildcard(require("react"));
var _propTypes = _interopRequireDefault(require("prop-types"));
var _RtlProvider = require("@mui/system/RtlProvider");
var _useId = _interopRequireDefault(require("@mui/utils/useId"));
var _warning = require("@mui/x-internals/warning");
var _gridClasses = require("../../constants/gridClasses");
var _GridMenu = require("../menu/GridMenu");
var _useGridRootProps = require("../../hooks/utils/useGridRootProps");
var _useGridApiContext = require("../../hooks/utils/useGridApiContext");
var _GridActionsCellItem = require("./GridActionsCellItem");
var _jsxRuntime = require("react/jsx-runtime");
const _excluded = ["api", "colDef", "id", "hasFocus", "isEditable", "field", "value", "formattedValue", "row", "rowNode", "cellMode", "tabIndex", "position", "onMenuOpen", "onMenuClose", "children", "suppressChildrenValidation"];
const hasActions = colDef => typeof colDef.getActions === 'function';
function GridActionsCell(props) {
  const {
      id,
      hasFocus,
      tabIndex,
      position = 'bottom-end',
      onMenuOpen,
      onMenuClose,
      children,
      suppressChildrenValidation
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded);
  const [focusedButtonIndex, setFocusedButtonIndex] = React.useState(-1);
  const [open, setOpen] = React.useState(false);
  const apiRef = (0, _useGridApiContext.useGridApiContext)();
  const rootRef = React.useRef(null);
  const buttonRef = React.useRef(null);
  const ignoreCallToFocus = React.useRef(false);
  const touchRippleRefs = React.useRef({});
  const isRtl = (0, _RtlProvider.useRtl)();
  const menuId = (0, _useId.default)();
  const buttonId = (0, _useId.default)();
  const rootProps = (0, _useGridRootProps.useGridRootProps)();
  const rowParams = apiRef.current.getRowParams(id);
  const actions = [];
  React.Children.forEach(children, child => {
    // Unwrap React.Fragment
    if (/*#__PURE__*/React.isValidElement(child)) {
      if (child.type === React.Fragment) {
        React.Children.forEach(child.props.children, fragChild => {
          if (/*#__PURE__*/React.isValidElement(fragChild)) {
            actions.push(fragChild);
          }
        });
      } else if (child.type === _GridActionsCellItem.GridActionsCellItem || suppressChildrenValidation) {
        actions.push(child);
      } else {
        const childType = typeof child.type === 'function' ? child.type.name : child.type;
        (0, _warning.warnOnce)(`MUI X: Invalid child type in \`GridActionsCell\`. Expected \`GridActionsCellItem\` or \`React.Fragment\`, got \`${childType}\`.
If this is intentional, you can suppress this warning by passing the \`suppressChildrenValidation\` prop to \`GridActionsCell\`.`, 'error');
      }
    }
  });
  const iconButtons = actions.filter(option => !option.props.showInMenu);
  const menuButtons = actions.filter(option => option.props.showInMenu);
  const numberOfButtons = iconButtons.length + (menuButtons.length ? 1 : 0);
  React.useLayoutEffect(() => {
    if (!hasFocus) {
      Object.entries(touchRippleRefs.current).forEach(([index, ref]) => {
        ref?.stop({}, () => {
          delete touchRippleRefs.current[index];
        });
      });
    }
  }, [hasFocus]);
  React.useEffect(() => {
    if (focusedButtonIndex < 0 || !rootRef.current) {
      return;
    }
    if (focusedButtonIndex >= rootRef.current.children.length) {
      return;
    }
    const child = rootRef.current.children[focusedButtonIndex];
    child.focus({
      preventScroll: true
    });
  }, [focusedButtonIndex]);
  const firstFocusableButtonIndex = actions.findIndex(o => !o.props.disabled);
  React.useEffect(() => {
    if (hasFocus && focusedButtonIndex === -1) {
      setFocusedButtonIndex(firstFocusableButtonIndex);
    }
    if (!hasFocus) {
      setFocusedButtonIndex(-1);
      ignoreCallToFocus.current = false;
    }
  }, [hasFocus, focusedButtonIndex, firstFocusableButtonIndex]);
  React.useEffect(() => {
    if (focusedButtonIndex >= numberOfButtons) {
      setFocusedButtonIndex(numberOfButtons - 1);
    }
  }, [focusedButtonIndex, numberOfButtons]);
  const showMenu = event => {
    if (onMenuOpen && !onMenuOpen(rowParams, event)) {
      return;
    }
    setOpen(true);
    setFocusedButtonIndex(numberOfButtons - 1);
    ignoreCallToFocus.current = true;
  };
  const hideMenu = event => {
    if (onMenuClose && !onMenuClose(rowParams, event)) {
      return;
    }
    setOpen(false);
  };
  const toggleMenu = event => {
    event.stopPropagation();
    event.preventDefault();
    if (open) {
      hideMenu(event);
    } else {
      showMenu(event);
    }
  };
  const handleTouchRippleRef = index => instance => {
    touchRippleRefs.current[index] = instance;
  };
  const handleButtonClick = (index, onClick) => event => {
    setFocusedButtonIndex(index);
    ignoreCallToFocus.current = true;
    if (onClick) {
      onClick(event);
    }
  };
  const handleRootKeyDown = event => {
    if (numberOfButtons <= 1) {
      return;
    }
    const getNewIndex = (index, direction) => {
      if (index < 0 || index > actions.length) {
        return index;
      }

      // for rtl mode we need to reverse the direction
      const rtlMod = isRtl ? -1 : 1;
      const indexMod = (direction === 'left' ? -1 : 1) * rtlMod;

      // if the button that should receive focus is disabled go one more step
      return actions[index + indexMod]?.props.disabled ? getNewIndex(index + indexMod, direction) : index + indexMod;
    };
    let newIndex = focusedButtonIndex;
    if (event.key === 'ArrowRight') {
      newIndex = getNewIndex(focusedButtonIndex, 'right');
    } else if (event.key === 'ArrowLeft') {
      newIndex = getNewIndex(focusedButtonIndex, 'left');
    }
    if (newIndex < 0 || newIndex >= numberOfButtons) {
      return; // We're already in the first or last item = do nothing and let the grid listen the event
    }
    if (newIndex !== focusedButtonIndex) {
      event.preventDefault(); // Prevent scrolling
      event.stopPropagation(); // Don't stop propagation for other keys, for example ArrowUp
      setFocusedButtonIndex(newIndex);
    }
  };

  // role="menu" requires at least one child element
  const attributes = numberOfButtons > 0 ? {
    role: 'menu',
    onKeyDown: handleRootKeyDown
  } : undefined;
  return /*#__PURE__*/(0, _jsxRuntime.jsxs)("div", (0, _extends2.default)({
    ref: rootRef,
    tabIndex: -1,
    className: _gridClasses.gridClasses.actionsCell
  }, attributes, other, {
    children: [iconButtons.map((button, index) => /*#__PURE__*/React.cloneElement(button, {
      key: index,
      touchRippleRef: handleTouchRippleRef(index),
      onClick: handleButtonClick(index, button.props.onClick),
      tabIndex: focusedButtonIndex === index ? tabIndex : -1
    })), menuButtons.length > 0 && buttonId && /*#__PURE__*/(0, _jsxRuntime.jsx)(rootProps.slots.baseIconButton, (0, _extends2.default)({
      ref: buttonRef,
      id: buttonId,
      "aria-label": apiRef.current.getLocaleText('actionsCellMore'),
      "aria-haspopup": "menu",
      "aria-expanded": open,
      "aria-controls": open ? menuId : undefined,
      role: "menuitem",
      size: "small",
      onClick: toggleMenu,
      touchRippleRef: handleTouchRippleRef(buttonId),
      tabIndex: focusedButtonIndex === iconButtons.length ? tabIndex : -1
    }, rootProps.slotProps?.baseIconButton, {
      children: /*#__PURE__*/(0, _jsxRuntime.jsx)(rootProps.slots.moreActionsIcon, {
        fontSize: "small"
      })
    })), menuButtons.length > 0 && /*#__PURE__*/(0, _jsxRuntime.jsx)(_GridMenu.GridMenu, {
      open: open,
      target: buttonRef.current,
      position: position,
      onClose: hideMenu,
      children: /*#__PURE__*/(0, _jsxRuntime.jsx)(rootProps.slots.baseMenuList, {
        id: menuId,
        className: _gridClasses.gridClasses.menuList,
        "aria-labelledby": buttonId,
        autoFocusItem: true,
        children: menuButtons.map((button, index) => /*#__PURE__*/React.cloneElement(button, {
          key: index,
          closeMenu: hideMenu
        }))
      })
    })]
  }));
}
process.env.NODE_ENV !== "production" ? GridActionsCell.propTypes = {
  // ----------------------------- Warning --------------------------------
  // | These PropTypes are generated from the TypeScript type definitions |
  // | To update them edit the TypeScript types and run "pnpm proptypes"  |
  // ----------------------------------------------------------------------
  api: _propTypes.default.object,
  /**
   * The mode of the cell.
   */
  cellMode: _propTypes.default.oneOf(['edit', 'view']).isRequired,
  children: _propTypes.default.node.isRequired,
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
  position: _propTypes.default.oneOf(['bottom-end', 'bottom-start', 'bottom', 'left-end', 'left-start', 'left', 'right-end', 'right-start', 'right', 'top-end', 'top-start', 'top']),
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
// Temporary wrapper for backward compatibility.
// Only used to support `getActions` method in `GridColDef`.
// TODO(v9): Remove this wrapper and the default `renderCell` in gridActionsColDef
function GridActionsCellWrapper(props) {
  const {
    colDef,
    id
  } = props;
  const apiRef = (0, _useGridApiContext.useGridApiContext)();
  if (!hasActions(colDef)) {
    throw new Error('MUI X: Missing the `getActions` property in the `GridColDef`.');
  }
  const actions = colDef.getActions(apiRef.current.getRowParams(id));
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(GridActionsCell, (0, _extends2.default)({
    suppressChildrenValidation: true
  }, props, {
    children: actions
  }));
}
const renderActionsCell = params => /*#__PURE__*/(0, _jsxRuntime.jsx)(GridActionsCellWrapper, (0, _extends2.default)({}, params));
exports.renderActionsCell = renderActionsCell;
if (process.env.NODE_ENV !== "production") renderActionsCell.displayName = "renderActionsCell";