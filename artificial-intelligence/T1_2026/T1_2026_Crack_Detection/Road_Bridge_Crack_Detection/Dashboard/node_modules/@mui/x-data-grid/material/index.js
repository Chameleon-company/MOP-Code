"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;
Object.defineProperty(exports, "useMaterialCSSVariables", {
  enumerable: true,
  get: function () {
    return _variables.useMaterialCSSVariables;
  }
});
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var _objectWithoutPropertiesLoose2 = _interopRequireDefault(require("@babel/runtime/helpers/objectWithoutPropertiesLoose"));
var React = _interopRequireWildcard(require("react"));
var _clsx = _interopRequireDefault(require("clsx"));
var _useForkRef = _interopRequireDefault(require("@mui/utils/useForkRef"));
var _useEventCallback = _interopRequireDefault(require("@mui/utils/useEventCallback"));
var _styles = require("@mui/material/styles");
var _Autocomplete = _interopRequireDefault(require("@mui/material/Autocomplete"));
var _Badge = _interopRequireDefault(require("@mui/material/Badge"));
var _Checkbox = _interopRequireDefault(require("@mui/material/Checkbox"));
var _Chip = _interopRequireDefault(require("@mui/material/Chip"));
var _CircularProgress = _interopRequireDefault(require("@mui/material/CircularProgress"));
var _Divider = _interopRequireDefault(require("@mui/material/Divider"));
var _InputBase = _interopRequireDefault(require("@mui/material/InputBase"));
var _Unstable_TrapFocus = _interopRequireDefault(require("@mui/material/Unstable_TrapFocus"));
var _LinearProgress = _interopRequireDefault(require("@mui/material/LinearProgress"));
var _ListItemIcon = _interopRequireDefault(require("@mui/material/ListItemIcon"));
var _ListItemText = _interopRequireWildcard(require("@mui/material/ListItemText"));
var _MenuList = _interopRequireDefault(require("@mui/material/MenuList"));
var _MenuItem = _interopRequireDefault(require("@mui/material/MenuItem"));
var _TextField = _interopRequireDefault(require("@mui/material/TextField"));
var _TextareaAutosize = _interopRequireDefault(require("@mui/material/TextareaAutosize"));
var _FormControl = _interopRequireDefault(require("@mui/material/FormControl"));
var _FormControlLabel = _interopRequireWildcard(require("@mui/material/FormControlLabel"));
var _Select = _interopRequireDefault(require("@mui/material/Select"));
var _Switch = _interopRequireDefault(require("@mui/material/Switch"));
var _Button = _interopRequireDefault(require("@mui/material/Button"));
var _IconButton = _interopRequireWildcard(require("@mui/material/IconButton"));
var _InputAdornment = _interopRequireWildcard(require("@mui/material/InputAdornment"));
var _Tooltip = _interopRequireDefault(require("@mui/material/Tooltip"));
var _TablePagination = _interopRequireWildcard(require("@mui/material/TablePagination"));
var _Popper = _interopRequireDefault(require("@mui/material/Popper"));
var _ClickAwayListener = _interopRequireDefault(require("@mui/material/ClickAwayListener"));
var _Grow = _interopRequireDefault(require("@mui/material/Grow"));
var _Paper = _interopRequireDefault(require("@mui/material/Paper"));
var _InputLabel = _interopRequireDefault(require("@mui/material/InputLabel"));
var _Skeleton = _interopRequireDefault(require("@mui/material/Skeleton"));
var _Tabs = _interopRequireDefault(require("@mui/material/Tabs"));
var _Tab = _interopRequireDefault(require("@mui/material/Tab"));
var _ToggleButton = _interopRequireDefault(require("@mui/material/ToggleButton"));
var _forwardRef = require("@mui/x-internals/forwardRef");
var _useId = _interopRequireDefault(require("@mui/utils/useId"));
var _icons = require("./icons");
var _GridColumnUnsortedIcon = require("../components/GridColumnUnsortedIcon");
var _useGridApiContext = require("../hooks/utils/useGridApiContext");
var _useGridRootProps = require("../hooks/utils/useGridRootProps");
require("./augmentation");
var _jsxRuntime = require("react/jsx-runtime");
var _variables = require("./variables");
const _excluded = ["id", "label", "labelId", "material", "disabled", "slotProps", "onChange", "onKeyDown", "onOpen", "onClose", "size", "style", "fullWidth"],
  _excluded2 = ["onRowsPerPageChange", "material", "disabled"],
  _excluded3 = ["material"],
  _excluded4 = ["autoFocus", "label", "fullWidth", "slotProps", "className", "material"],
  _excluded5 = ["material"],
  _excluded6 = ["material"],
  _excluded7 = ["material"],
  _excluded8 = ["material"],
  _excluded9 = ["material"],
  _excluded0 = ["material"],
  _excluded1 = ["material"],
  _excluded10 = ["material"],
  _excluded11 = ["material"],
  _excluded12 = ["material", "label", "className"],
  _excluded13 = ["material"],
  _excluded14 = ["inert", "iconStart", "iconEnd", "children", "material"],
  _excluded15 = ["slotProps", "material"],
  _excluded16 = ["id", "multiple", "freeSolo", "options", "getOptionLabel", "isOptionEqualToValue", "value", "onChange", "label", "placeholder", "slotProps", "material"],
  _excluded17 = ["key"],
  _excluded18 = ["inputProps", "InputProps", "InputLabelProps"],
  _excluded19 = ["slotProps", "material"],
  _excluded20 = ["material"],
  _excluded21 = ["ref", "open", "children", "className", "clickAwayTouchEvent", "clickAwayMouseEvent", "flip", "focusTrap", "onExited", "onClickAway", "onDidShow", "onDidHide", "id", "target", "transition", "placement", "material"],
  _excluded22 = ["native"],
  _excluded23 = ["children", "value", "active"],
  _excluded24 = ["items", "value", "material"];
/* eslint-disable material-ui/disallow-react-api-in-server-components */

const InputAdornment = (0, _styles.styled)(_InputAdornment.default, {
  slot: 'internal'
})(({
  theme
}) => ({
  [`&.${_InputAdornment.inputAdornmentClasses.positionEnd} .${_IconButton.iconButtonClasses.sizeSmall}`]: {
    marginRight: theme.spacing(-0.75)
  }
}));
const FormControlLabel = (0, _styles.styled)(_FormControlLabel.default, {
  slot: 'internal',
  shouldForwardProp: prop => prop !== 'fullWidth'
})(({
  theme
}) => ({
  gap: theme.spacing(0.5),
  margin: 0,
  overflow: 'hidden',
  [`& .${_FormControlLabel.formControlLabelClasses.label}`]: {
    fontSize: theme.typography.pxToRem(14),
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap'
  },
  variants: [{
    props: {
      fullWidth: true
    },
    style: {
      width: '100%'
    }
  }]
}));
const Checkbox = (0, _styles.styled)(_Checkbox.default, {
  slot: 'internal',
  shouldForwardProp: prop => prop !== 'density'
})(({
  theme
}) => ({
  variants: [{
    props: {
      density: 'compact'
    },
    style: {
      padding: theme.spacing(0.5)
    }
  }]
}));
const ListItemText = (0, _styles.styled)(_ListItemText.default, {
  slot: 'internal'
})({
  [`& .${_ListItemText.listItemTextClasses.primary}`]: {
    overflowX: 'clip',
    textOverflow: 'ellipsis',
    maxWidth: '300px'
  }
});
const BaseSelect = (0, _forwardRef.forwardRef)(function BaseSelect(props, ref) {
  const {
      id,
      label,
      labelId,
      material,
      disabled,
      slotProps,
      onChange,
      onKeyDown,
      onOpen,
      onClose,
      size,
      style,
      fullWidth
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded);
  const theme = (0, _styles.useTheme)();
  const textFieldDefaults = theme.components?.MuiTextField?.defaultProps ?? {};
  const computedSize = size ?? textFieldDefaults.size;
  const computedVariant = textFieldDefaults.variant ?? 'outlined';
  const menuProps = {
    PaperProps: {
      onKeyDown
    }
  };
  if (onClose) {
    menuProps.onClose = onClose;
  }
  return /*#__PURE__*/(0, _jsxRuntime.jsxs)(_FormControl.default, {
    size: computedSize,
    fullWidth: fullWidth,
    style: style,
    disabled: disabled,
    ref: ref,
    children: [/*#__PURE__*/(0, _jsxRuntime.jsx)(_InputLabel.default, {
      id: labelId,
      htmlFor: id,
      shrink: true,
      variant: computedVariant,
      children: label
    }), /*#__PURE__*/(0, _jsxRuntime.jsx)(_Select.default, (0, _extends2.default)({
      id: id,
      labelId: labelId,
      label: label,
      displayEmpty: true,
      onChange: onChange,
      variant: computedVariant
    }, other, {
      inputProps: slotProps?.htmlInput,
      onOpen: onOpen,
      MenuProps: menuProps,
      size: computedSize
    }, material))]
  });
});
if (process.env.NODE_ENV !== "production") BaseSelect.displayName = "BaseSelect";
const StyledPagination = (0, _styles.styled)(_TablePagination.default, {
  slot: 'internal'
})(({
  theme
}) => ({
  [`& .${_TablePagination.tablePaginationClasses.selectLabel}`]: {
    display: 'none',
    [theme.breakpoints.up('sm')]: {
      display: 'block'
    }
  },
  [`& .${_TablePagination.tablePaginationClasses.input}`]: {
    display: 'none',
    [theme.breakpoints.up('sm')]: {
      display: 'inline-flex'
    }
  }
}));
const BasePagination = (0, _forwardRef.forwardRef)(function BasePagination(props, ref) {
  const {
      onRowsPerPageChange,
      material,
      disabled
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded2);
  const computedProps = React.useMemo(() => {
    if (!disabled) {
      return undefined;
    }
    return {
      backIconButtonProps: {
        disabled: true
      },
      nextIconButtonProps: {
        disabled: true
      }
    };
  }, [disabled]);
  const apiRef = (0, _useGridApiContext.useGridApiContext)();
  const rootProps = (0, _useGridRootProps.useGridRootProps)();
  const {
    estimatedRowCount
  } = rootProps;
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(StyledPagination, (0, _extends2.default)({
    component: "div",
    onRowsPerPageChange: (0, _useEventCallback.default)(event => {
      onRowsPerPageChange?.(Number(event.target.value));
    }),
    labelRowsPerPage: apiRef.current.getLocaleText('paginationRowsPerPage'),
    labelDisplayedRows: params => apiRef.current.getLocaleText('paginationDisplayedRows')((0, _extends2.default)({}, params, {
      estimated: estimatedRowCount
    })),
    getItemAriaLabel: apiRef.current.getLocaleText('paginationItemAriaLabel')
  }, computedProps, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BasePagination.displayName = "BasePagination";
const BaseBadge = (0, _forwardRef.forwardRef)(function BaseBadge(props, ref) {
  const {
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded3);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_Badge.default, (0, _extends2.default)({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseBadge.displayName = "BaseBadge";
const BaseCheckbox = (0, _forwardRef.forwardRef)(function BaseCheckbox(props, ref) {
  const {
      autoFocus,
      label,
      fullWidth,
      slotProps,
      className,
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded4);
  const elementRef = React.useRef(null);
  const handleRef = (0, _useForkRef.default)(elementRef, ref);
  const rippleRef = React.useRef(null);
  React.useEffect(() => {
    if (autoFocus) {
      const input = elementRef.current?.querySelector('input');
      input?.focus({
        preventScroll: true
      });
    } else if (autoFocus === false && rippleRef.current) {
      // Only available in @mui/material v5.4.1 or later
      // @ts-ignore
      rippleRef.current.stop({});
    }
  }, [autoFocus]);
  if (!label) {
    return /*#__PURE__*/(0, _jsxRuntime.jsx)(Checkbox, (0, _extends2.default)({}, other, material, {
      className: (0, _clsx.default)(className, material?.className),
      inputProps: slotProps?.htmlInput,
      ref: handleRef,
      touchRippleRef: rippleRef
    }));
  }
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(FormControlLabel, {
    className: className,
    control: /*#__PURE__*/(0, _jsxRuntime.jsx)(Checkbox, (0, _extends2.default)({}, other, material, {
      inputProps: slotProps?.htmlInput,
      ref: handleRef,
      touchRippleRef: rippleRef
    })),
    label: label,
    fullWidth: fullWidth
  });
});
if (process.env.NODE_ENV !== "production") BaseCheckbox.displayName = "BaseCheckbox";
const BaseCircularProgress = (0, _forwardRef.forwardRef)(function BaseCircularProgress(props, ref) {
  const {
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded5);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_CircularProgress.default, (0, _extends2.default)({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseCircularProgress.displayName = "BaseCircularProgress";
const BaseDivider = (0, _forwardRef.forwardRef)(function BaseDivider(props, ref) {
  const {
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded6);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_Divider.default, (0, _extends2.default)({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseDivider.displayName = "BaseDivider";
const BaseLinearProgress = (0, _forwardRef.forwardRef)(function BaseLinearProgress(props, ref) {
  const {
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded7);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_LinearProgress.default, (0, _extends2.default)({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseLinearProgress.displayName = "BaseLinearProgress";
const BaseButton = (0, _forwardRef.forwardRef)(function BaseButton(props, ref) {
  const {
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded8);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_Button.default, (0, _extends2.default)({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseButton.displayName = "BaseButton";
const StyledToggleButton = (0, _styles.styled)(_ToggleButton.default, {
  slot: 'internal'
})(({
  theme
}) => ({
  gap: theme.spacing(1),
  border: 0
}));
const BaseToggleButton = (0, _forwardRef.forwardRef)(function BaseToggleButton(props, ref) {
  const {
      material
    } = props,
    rest = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded9);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(StyledToggleButton, (0, _extends2.default)({
    size: "small",
    color: "primary"
  }, rest, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseToggleButton.displayName = "BaseToggleButton";
const BaseChip = (0, _forwardRef.forwardRef)(function BaseChip(props, ref) {
  const {
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded0);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_Chip.default, (0, _extends2.default)({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseChip.displayName = "BaseChip";
const BaseIconButton = (0, _forwardRef.forwardRef)(function BaseIconButton(props, ref) {
  const {
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded1);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_IconButton.default, (0, _extends2.default)({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseIconButton.displayName = "BaseIconButton";
const BaseTooltip = (0, _forwardRef.forwardRef)(function BaseTooltip(props, ref) {
  const {
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded10);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_Tooltip.default, (0, _extends2.default)({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseTooltip.displayName = "BaseTooltip";
const BaseSkeleton = (0, _forwardRef.forwardRef)(function BaseSkeleton(props, ref) {
  const {
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded11);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_Skeleton.default, (0, _extends2.default)({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseSkeleton.displayName = "BaseSkeleton";
const BaseSwitch = (0, _forwardRef.forwardRef)(function BaseSwitch(props, ref) {
  const {
      material,
      label,
      className
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded12);
  if (!label) {
    return /*#__PURE__*/(0, _jsxRuntime.jsx)(_Switch.default, (0, _extends2.default)({}, other, material, {
      className: className,
      ref: ref
    }));
  }
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(FormControlLabel, {
    className: className,
    control: /*#__PURE__*/(0, _jsxRuntime.jsx)(_Switch.default, (0, _extends2.default)({}, other, material, {
      ref: ref
    })),
    label: label
  });
});
if (process.env.NODE_ENV !== "production") BaseSwitch.displayName = "BaseSwitch";
const BaseMenuList = (0, _forwardRef.forwardRef)(function BaseMenuList(props, ref) {
  const {
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded13);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_MenuList.default, (0, _extends2.default)({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseMenuList.displayName = "BaseMenuList";
function BaseMenuItem(props) {
  const {
      inert,
      iconStart,
      iconEnd,
      children,
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded14);
  if (inert) {
    other.disableRipple = true;
  }
  return /*#__PURE__*/React.createElement(_MenuItem.default, (0, _extends2.default)({}, other, material), [iconStart && /*#__PURE__*/(0, _jsxRuntime.jsx)(_ListItemIcon.default, {
    children: iconStart
  }, "1"), /*#__PURE__*/(0, _jsxRuntime.jsx)(ListItemText, {
    children: children
  }, "2"), iconEnd && /*#__PURE__*/(0, _jsxRuntime.jsx)(_ListItemIcon.default, {
    children: iconEnd
  }, "3")]);
}
function BaseTextField(props) {
  // MaterialUI v5 doesn't support slotProps, until we drop v5 support we need to
  // translate the pattern.
  const {
      slotProps,
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded15);
  const theme = (0, _styles.useTheme)();
  const textFieldDefaults = theme.components?.MuiTextField?.defaultProps ?? {};
  const computedVariant = other.variant ?? textFieldDefaults.variant ?? 'outlined';
  const computedSize = other.size ?? textFieldDefaults.size;
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_TextField.default, (0, _extends2.default)({
    variant: computedVariant,
    size: computedSize
  }, other, material, {
    inputProps: slotProps?.htmlInput,
    InputProps: transformInputProps(slotProps?.input),
    InputLabelProps: (0, _extends2.default)({
      shrink: true
    }, slotProps?.inputLabel)
  }));
}
function BaseAutocomplete(props) {
  const rootProps = (0, _useGridRootProps.useGridRootProps)();
  const {
      id,
      multiple,
      freeSolo,
      options,
      getOptionLabel,
      isOptionEqualToValue,
      value,
      onChange,
      label,
      placeholder,
      slotProps,
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded16);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_Autocomplete.default, (0, _extends2.default)({
    id: id,
    multiple: multiple,
    freeSolo: freeSolo,
    options: options,
    getOptionLabel: getOptionLabel,
    isOptionEqualToValue: isOptionEqualToValue,
    value: value,
    onChange: onChange,
    renderTags: (currentValue, getTagProps) => currentValue.map((option, index) => {
      const _getTagProps = getTagProps({
          index
        }),
        {
          key
        } = _getTagProps,
        tagProps = (0, _objectWithoutPropertiesLoose2.default)(_getTagProps, _excluded17);
      return /*#__PURE__*/(0, _jsxRuntime.jsx)(_Chip.default, (0, _extends2.default)({
        variant: "outlined",
        size: "small",
        label: typeof option === 'string' ? option : getOptionLabel?.(option)
      }, tagProps), key);
    }),
    renderInput: params => {
      const {
          inputProps,
          InputProps,
          InputLabelProps
        } = params,
        inputRest = (0, _objectWithoutPropertiesLoose2.default)(params, _excluded18);
      return /*#__PURE__*/(0, _jsxRuntime.jsx)(_TextField.default, (0, _extends2.default)({}, inputRest, {
        label: label,
        placeholder: placeholder,
        inputProps: inputProps,
        InputProps: transformInputProps(InputProps, false),
        InputLabelProps: (0, _extends2.default)({
          shrink: true
        }, InputLabelProps)
      }, slotProps?.textField, rootProps.slotProps?.baseTextField));
    }
  }, other, material));
}
function BaseInput(props) {
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_InputBase.default, (0, _extends2.default)({}, transformInputProps(props)));
}
function transformInputProps(props, wrapAdornments = true) {
  if (!props) {
    return undefined;
  }
  const {
      slotProps,
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded19);
  const result = other;
  if (wrapAdornments) {
    if (result.startAdornment) {
      result.startAdornment = /*#__PURE__*/(0, _jsxRuntime.jsx)(InputAdornment, {
        position: "start",
        children: result.startAdornment
      });
    }
    if (result.endAdornment) {
      result.endAdornment = /*#__PURE__*/(0, _jsxRuntime.jsx)(InputAdornment, {
        position: "end",
        children: result.endAdornment
      });
    }
  }
  for (const k in material) {
    if (Object.hasOwn(material, k)) {
      result[k] = material[k];
    }
  }
  if (slotProps?.htmlInput) {
    if (result.inputProps) {
      result.inputProps = (0, _extends2.default)({}, result.inputProps, slotProps?.htmlInput);
    } else {
      result.inputProps = slotProps?.htmlInput;
    }
  }
  return result;
}
const BaseTextarea = (0, _forwardRef.forwardRef)(function BaseTextarea(props, ref) {
  const {
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded20);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_TextareaAutosize.default, (0, _extends2.default)({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseTextarea.displayName = "BaseTextarea";
const transformOrigin = {
  'bottom-start': 'top left',
  'bottom-end': 'top right'
};
function BasePopper(props) {
  const {
      open,
      children,
      className,
      flip,
      onExited,
      onDidShow,
      onDidHide,
      id,
      target,
      transition,
      placement,
      material
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded21);
  const modifiers = React.useMemo(() => {
    const result = [{
      name: 'preventOverflow',
      options: {
        padding: 8
      }
    }];
    if (flip) {
      result.push({
        name: 'flip',
        enabled: true
      });
    }
    if (onDidShow || onDidHide) {
      result.push({
        name: 'isPlaced',
        enabled: true,
        phase: 'main',
        fn: () => {
          onDidShow?.();
        },
        effect: () => () => {
          onDidHide?.();
        }
      });
    }
    return result;
  }, [flip, onDidShow, onDidHide]);
  let content;
  if (!transition) {
    content = wrappers(props, children);
  } else {
    const handleExited = popperOnExited => node => {
      if (popperOnExited) {
        popperOnExited();
      }
      if (onExited) {
        onExited(node);
      }
    };
    content = p => wrappers(props, /*#__PURE__*/(0, _jsxRuntime.jsx)(_Grow.default, (0, _extends2.default)({}, p.TransitionProps, {
      style: {
        transformOrigin: transformOrigin[p.placement]
      },
      onExited: handleExited(p.TransitionProps?.onExited),
      children: /*#__PURE__*/(0, _jsxRuntime.jsx)(_Paper.default, {
        children: children
      })
    })));
  }
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_Popper.default, (0, _extends2.default)({
    id: id,
    className: className,
    open: open,
    anchorEl: target,
    transition: transition,
    placement: placement,
    modifiers: modifiers
  }, other, material, {
    children: content
  }));
}
function wrappers(props, content) {
  return focusTrapWrapper(props, clickAwayWrapper(props, content));
}
function clickAwayWrapper(props, content) {
  if (props.onClickAway === undefined) {
    return content;
  }
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_ClickAwayListener.default, {
    onClickAway: props.onClickAway,
    touchEvent: props.clickAwayTouchEvent,
    mouseEvent: props.clickAwayMouseEvent,
    children: content
  });
}
function focusTrapWrapper(props, content) {
  if (props.focusTrap === undefined) {
    return content;
  }
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_Unstable_TrapFocus.default, {
    open: true,
    disableEnforceFocus: true,
    disableAutoFocus: true,
    children: /*#__PURE__*/(0, _jsxRuntime.jsx)("div", {
      tabIndex: -1,
      children: content
    })
  });
}
function BaseSelectOption(_ref) {
  let {
      native
    } = _ref,
    props = (0, _objectWithoutPropertiesLoose2.default)(_ref, _excluded22);
  if (native) {
    return /*#__PURE__*/(0, _jsxRuntime.jsx)("option", (0, _extends2.default)({}, props));
  }
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_MenuItem.default, (0, _extends2.default)({}, props));
}
const StyledTabs = (0, _styles.styled)(_Tabs.default, {
  name: 'MuiDataGrid',
  slot: 'Tabs'
})(({
  theme
}) => ({
  borderBottom: `1px solid ${theme.palette.divider}`
}));
const StyledTab = (0, _styles.styled)(_Tab.default, {
  name: 'MuiDataGrid',
  slot: 'Tab'
})({
  flex: 1,
  minWidth: 'fit-content'
});
const StyledTabPanel = (0, _styles.styled)('div', {
  name: 'MuiDataGrid',
  slot: 'TabPanel'
})({
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden'
});
function TabPanel(props) {
  const {
      children,
      active
    } = props,
    other = (0, _objectWithoutPropertiesLoose2.default)(props, _excluded23);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(StyledTabPanel, (0, _extends2.default)({
    role: "tabpanel",
    style: {
      display: active ? 'flex' : 'none'
    }
  }, other, {
    children: children
  }));
}
function BaseTabs(_ref2) {
  let {
      items,
      value,
      material
    } = _ref2,
    props = (0, _objectWithoutPropertiesLoose2.default)(_ref2, _excluded24);
  const id = (0, _useId.default)();
  const labelId = `${id}-tab-${value}`;
  const panelId = `${id}-tabpanel-${value}`;
  return /*#__PURE__*/(0, _jsxRuntime.jsxs)(React.Fragment, {
    children: [/*#__PURE__*/(0, _jsxRuntime.jsx)(StyledTabs, (0, _extends2.default)({}, props, {
      value: value,
      variant: "scrollable",
      scrollButtons: "auto"
    }, material, {
      children: items.map(item => /*#__PURE__*/(0, _jsxRuntime.jsx)(StyledTab, {
        value: item.value,
        label: item.label,
        id: labelId,
        "aria-controls": panelId
      }, item.value))
    })), items.map(item => /*#__PURE__*/(0, _jsxRuntime.jsx)(TabPanel, {
      value: item.value,
      active: value === item.value,
      id: panelId,
      "aria-labelledby": labelId,
      children: item.children
    }, item.value))]
  });
}
const iconSlots = {
  booleanCellTrueIcon: _icons.GridCheckIcon,
  booleanCellFalseIcon: _icons.GridCloseIcon,
  columnMenuIcon: _icons.GridTripleDotsVerticalIcon,
  openFilterButtonIcon: _icons.GridFilterListIcon,
  filterPanelDeleteIcon: _icons.GridCloseIcon,
  undoIcon: _icons.GridUndoIcon,
  redoIcon: _icons.GridRedoIcon,
  columnFilteredIcon: _icons.GridFilterAltIcon,
  columnSelectorIcon: _icons.GridColumnIcon,
  columnUnsortedIcon: _GridColumnUnsortedIcon.GridColumnUnsortedIcon,
  columnSortedAscendingIcon: _icons.GridArrowUpwardIcon,
  columnSortedDescendingIcon: _icons.GridArrowDownwardIcon,
  columnResizeIcon: _icons.GridSeparatorIcon,
  densityCompactIcon: _icons.GridViewHeadlineIcon,
  densityStandardIcon: _icons.GridTableRowsIcon,
  densityComfortableIcon: _icons.GridViewStreamIcon,
  exportIcon: _icons.GridDownloadIcon,
  moreActionsIcon: _icons.GridMoreVertIcon,
  treeDataCollapseIcon: _icons.GridExpandMoreIcon,
  treeDataExpandIcon: _icons.GridKeyboardArrowRight,
  groupingCriteriaCollapseIcon: _icons.GridExpandMoreIcon,
  groupingCriteriaExpandIcon: _icons.GridKeyboardArrowRight,
  detailPanelExpandIcon: _icons.GridAddIcon,
  detailPanelCollapseIcon: _icons.GridRemoveIcon,
  rowReorderIcon: _icons.GridDragIcon,
  quickFilterIcon: _icons.GridSearchIcon,
  quickFilterClearIcon: _icons.GridClearIcon,
  columnMenuHideIcon: _icons.GridVisibilityOffIcon,
  columnMenuSortAscendingIcon: _icons.GridArrowUpwardIcon,
  columnMenuSortDescendingIcon: _icons.GridArrowDownwardIcon,
  columnMenuUnsortIcon: null,
  columnMenuFilterIcon: _icons.GridFilterAltIcon,
  columnMenuManageColumnsIcon: _icons.GridViewColumnIcon,
  columnMenuClearIcon: _icons.GridClearIcon,
  loadIcon: _icons.GridLoadIcon,
  filterPanelAddIcon: _icons.GridAddIcon,
  filterPanelRemoveAllIcon: _icons.GridDeleteForeverIcon,
  columnReorderIcon: _icons.GridDragIcon,
  menuItemCheckIcon: _icons.GridCheckIcon,
  longTextCellExpandIcon: _icons.GridLongTextCellExpandIcon,
  longTextCellCollapseIcon: _icons.GridLongTextCellCollapseIcon
};
const baseSlots = {
  baseAutocomplete: BaseAutocomplete,
  baseBadge: BaseBadge,
  baseCheckbox: BaseCheckbox,
  baseChip: BaseChip,
  baseCircularProgress: BaseCircularProgress,
  baseDivider: BaseDivider,
  baseInput: BaseInput,
  baseTextarea: BaseTextarea,
  baseLinearProgress: BaseLinearProgress,
  baseMenuList: BaseMenuList,
  baseMenuItem: BaseMenuItem,
  baseTextField: BaseTextField,
  baseButton: BaseButton,
  baseIconButton: BaseIconButton,
  baseToggleButton: BaseToggleButton,
  baseTooltip: BaseTooltip,
  baseTabs: BaseTabs,
  basePagination: BasePagination,
  basePopper: BasePopper,
  baseSelect: BaseSelect,
  baseSelectOption: BaseSelectOption,
  baseSkeleton: BaseSkeleton,
  baseSwitch: BaseSwitch
};
const materialSlots = (0, _extends2.default)({}, baseSlots, iconSlots);
var _default = exports.default = materialSlots;