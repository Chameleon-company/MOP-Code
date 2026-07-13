import _extends from "@babel/runtime/helpers/esm/extends";
import _objectWithoutPropertiesLoose from "@babel/runtime/helpers/esm/objectWithoutPropertiesLoose";
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
import * as React from 'react';
import clsx from 'clsx';
import useForkRef from '@mui/utils/useForkRef';
import useEventCallback from '@mui/utils/useEventCallback';
import { styled, useTheme } from '@mui/material/styles';
import MUIAutocomplete from '@mui/material/Autocomplete';
import MUIBadge from '@mui/material/Badge';
import MUICheckbox from '@mui/material/Checkbox';
import MUIChip from '@mui/material/Chip';
import MUICircularProgress from '@mui/material/CircularProgress';
import MUIDivider from '@mui/material/Divider';
import MUIInputBase from '@mui/material/InputBase';
import MUIFocusTrap from '@mui/material/Unstable_TrapFocus';
import MUILinearProgress from '@mui/material/LinearProgress';
import MUIListItemIcon from '@mui/material/ListItemIcon';
import MUIListItemText, { listItemTextClasses } from '@mui/material/ListItemText';
import MUIMenuList from '@mui/material/MenuList';
import MUIMenuItem from '@mui/material/MenuItem';
import MUITextField from '@mui/material/TextField';
import MUITextareaAutosize from '@mui/material/TextareaAutosize';
import MUIFormControl from '@mui/material/FormControl';
import MUIFormControlLabel, { formControlLabelClasses } from '@mui/material/FormControlLabel';
import MUISelect from '@mui/material/Select';
import MUISwitch from '@mui/material/Switch';
import MUIButton from '@mui/material/Button';
import MUIIconButton, { iconButtonClasses } from '@mui/material/IconButton';
import MUIInputAdornment, { inputAdornmentClasses } from '@mui/material/InputAdornment';
import MUITooltip from '@mui/material/Tooltip';
import MUIPagination, { tablePaginationClasses } from '@mui/material/TablePagination';
import MUIPopper from '@mui/material/Popper';
import ClickAwayListener from '@mui/material/ClickAwayListener';
import MUIGrow from '@mui/material/Grow';
import MUIPaper from '@mui/material/Paper';
import MUIInputLabel from '@mui/material/InputLabel';
import MUISkeleton from '@mui/material/Skeleton';
import MUITabs from '@mui/material/Tabs';
import MUITab from '@mui/material/Tab';
import MUIToggleButton from '@mui/material/ToggleButton';
import { forwardRef } from '@mui/x-internals/forwardRef';
import useId from '@mui/utils/useId';
import { GridAddIcon, GridArrowDownwardIcon, GridArrowUpwardIcon, GridCheckIcon, GridCloseIcon, GridUndoIcon, GridRedoIcon, GridColumnIcon, GridDragIcon, GridExpandMoreIcon, GridFilterAltIcon, GridFilterListIcon, GridKeyboardArrowRight, GridMoreVertIcon, GridRemoveIcon, GridSearchIcon, GridSeparatorIcon, GridTableRowsIcon, GridTripleDotsVerticalIcon, GridViewHeadlineIcon, GridViewStreamIcon, GridVisibilityOffIcon, GridViewColumnIcon, GridClearIcon, GridLoadIcon, GridDeleteForeverIcon, GridDownloadIcon, GridLongTextCellExpandIcon, GridLongTextCellCollapseIcon } from "./icons/index.js";
import { GridColumnUnsortedIcon } from "../components/GridColumnUnsortedIcon.js";
import { useGridApiContext } from "../hooks/utils/useGridApiContext.js";
import { useGridRootProps } from "../hooks/utils/useGridRootProps.js";
import "./augmentation.js";
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export { useMaterialCSSVariables } from "./variables.js";

/* eslint-disable material-ui/disallow-react-api-in-server-components */

const InputAdornment = styled(MUIInputAdornment, {
  slot: 'internal'
})(({
  theme
}) => ({
  [`&.${inputAdornmentClasses.positionEnd} .${iconButtonClasses.sizeSmall}`]: {
    marginRight: theme.spacing(-0.75)
  }
}));
const FormControlLabel = styled(MUIFormControlLabel, {
  slot: 'internal',
  shouldForwardProp: prop => prop !== 'fullWidth'
})(({
  theme
}) => ({
  gap: theme.spacing(0.5),
  margin: 0,
  overflow: 'hidden',
  [`& .${formControlLabelClasses.label}`]: {
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
const Checkbox = styled(MUICheckbox, {
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
const ListItemText = styled(MUIListItemText, {
  slot: 'internal'
})({
  [`& .${listItemTextClasses.primary}`]: {
    overflowX: 'clip',
    textOverflow: 'ellipsis',
    maxWidth: '300px'
  }
});
const BaseSelect = forwardRef(function BaseSelect(props, ref) {
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
    other = _objectWithoutPropertiesLoose(props, _excluded);
  const theme = useTheme();
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
  return /*#__PURE__*/_jsxs(MUIFormControl, {
    size: computedSize,
    fullWidth: fullWidth,
    style: style,
    disabled: disabled,
    ref: ref,
    children: [/*#__PURE__*/_jsx(MUIInputLabel, {
      id: labelId,
      htmlFor: id,
      shrink: true,
      variant: computedVariant,
      children: label
    }), /*#__PURE__*/_jsx(MUISelect, _extends({
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
const StyledPagination = styled(MUIPagination, {
  slot: 'internal'
})(({
  theme
}) => ({
  [`& .${tablePaginationClasses.selectLabel}`]: {
    display: 'none',
    [theme.breakpoints.up('sm')]: {
      display: 'block'
    }
  },
  [`& .${tablePaginationClasses.input}`]: {
    display: 'none',
    [theme.breakpoints.up('sm')]: {
      display: 'inline-flex'
    }
  }
}));
const BasePagination = forwardRef(function BasePagination(props, ref) {
  const {
      onRowsPerPageChange,
      material,
      disabled
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded2);
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
  const apiRef = useGridApiContext();
  const rootProps = useGridRootProps();
  const {
    estimatedRowCount
  } = rootProps;
  return /*#__PURE__*/_jsx(StyledPagination, _extends({
    component: "div",
    onRowsPerPageChange: useEventCallback(event => {
      onRowsPerPageChange?.(Number(event.target.value));
    }),
    labelRowsPerPage: apiRef.current.getLocaleText('paginationRowsPerPage'),
    labelDisplayedRows: params => apiRef.current.getLocaleText('paginationDisplayedRows')(_extends({}, params, {
      estimated: estimatedRowCount
    })),
    getItemAriaLabel: apiRef.current.getLocaleText('paginationItemAriaLabel')
  }, computedProps, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BasePagination.displayName = "BasePagination";
const BaseBadge = forwardRef(function BaseBadge(props, ref) {
  const {
      material
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded3);
  return /*#__PURE__*/_jsx(MUIBadge, _extends({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseBadge.displayName = "BaseBadge";
const BaseCheckbox = forwardRef(function BaseCheckbox(props, ref) {
  const {
      autoFocus,
      label,
      fullWidth,
      slotProps,
      className,
      material
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded4);
  const elementRef = React.useRef(null);
  const handleRef = useForkRef(elementRef, ref);
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
    return /*#__PURE__*/_jsx(Checkbox, _extends({}, other, material, {
      className: clsx(className, material?.className),
      inputProps: slotProps?.htmlInput,
      ref: handleRef,
      touchRippleRef: rippleRef
    }));
  }
  return /*#__PURE__*/_jsx(FormControlLabel, {
    className: className,
    control: /*#__PURE__*/_jsx(Checkbox, _extends({}, other, material, {
      inputProps: slotProps?.htmlInput,
      ref: handleRef,
      touchRippleRef: rippleRef
    })),
    label: label,
    fullWidth: fullWidth
  });
});
if (process.env.NODE_ENV !== "production") BaseCheckbox.displayName = "BaseCheckbox";
const BaseCircularProgress = forwardRef(function BaseCircularProgress(props, ref) {
  const {
      material
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded5);
  return /*#__PURE__*/_jsx(MUICircularProgress, _extends({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseCircularProgress.displayName = "BaseCircularProgress";
const BaseDivider = forwardRef(function BaseDivider(props, ref) {
  const {
      material
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded6);
  return /*#__PURE__*/_jsx(MUIDivider, _extends({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseDivider.displayName = "BaseDivider";
const BaseLinearProgress = forwardRef(function BaseLinearProgress(props, ref) {
  const {
      material
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded7);
  return /*#__PURE__*/_jsx(MUILinearProgress, _extends({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseLinearProgress.displayName = "BaseLinearProgress";
const BaseButton = forwardRef(function BaseButton(props, ref) {
  const {
      material
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded8);
  return /*#__PURE__*/_jsx(MUIButton, _extends({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseButton.displayName = "BaseButton";
const StyledToggleButton = styled(MUIToggleButton, {
  slot: 'internal'
})(({
  theme
}) => ({
  gap: theme.spacing(1),
  border: 0
}));
const BaseToggleButton = forwardRef(function BaseToggleButton(props, ref) {
  const {
      material
    } = props,
    rest = _objectWithoutPropertiesLoose(props, _excluded9);
  return /*#__PURE__*/_jsx(StyledToggleButton, _extends({
    size: "small",
    color: "primary"
  }, rest, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseToggleButton.displayName = "BaseToggleButton";
const BaseChip = forwardRef(function BaseChip(props, ref) {
  const {
      material
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded0);
  return /*#__PURE__*/_jsx(MUIChip, _extends({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseChip.displayName = "BaseChip";
const BaseIconButton = forwardRef(function BaseIconButton(props, ref) {
  const {
      material
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded1);
  return /*#__PURE__*/_jsx(MUIIconButton, _extends({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseIconButton.displayName = "BaseIconButton";
const BaseTooltip = forwardRef(function BaseTooltip(props, ref) {
  const {
      material
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded10);
  return /*#__PURE__*/_jsx(MUITooltip, _extends({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseTooltip.displayName = "BaseTooltip";
const BaseSkeleton = forwardRef(function BaseSkeleton(props, ref) {
  const {
      material
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded11);
  return /*#__PURE__*/_jsx(MUISkeleton, _extends({}, other, material, {
    ref: ref
  }));
});
if (process.env.NODE_ENV !== "production") BaseSkeleton.displayName = "BaseSkeleton";
const BaseSwitch = forwardRef(function BaseSwitch(props, ref) {
  const {
      material,
      label,
      className
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded12);
  if (!label) {
    return /*#__PURE__*/_jsx(MUISwitch, _extends({}, other, material, {
      className: className,
      ref: ref
    }));
  }
  return /*#__PURE__*/_jsx(FormControlLabel, {
    className: className,
    control: /*#__PURE__*/_jsx(MUISwitch, _extends({}, other, material, {
      ref: ref
    })),
    label: label
  });
});
if (process.env.NODE_ENV !== "production") BaseSwitch.displayName = "BaseSwitch";
const BaseMenuList = forwardRef(function BaseMenuList(props, ref) {
  const {
      material
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded13);
  return /*#__PURE__*/_jsx(MUIMenuList, _extends({}, other, material, {
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
    other = _objectWithoutPropertiesLoose(props, _excluded14);
  if (inert) {
    other.disableRipple = true;
  }
  return /*#__PURE__*/React.createElement(MUIMenuItem, _extends({}, other, material), [iconStart && /*#__PURE__*/_jsx(MUIListItemIcon, {
    children: iconStart
  }, "1"), /*#__PURE__*/_jsx(ListItemText, {
    children: children
  }, "2"), iconEnd && /*#__PURE__*/_jsx(MUIListItemIcon, {
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
    other = _objectWithoutPropertiesLoose(props, _excluded15);
  const theme = useTheme();
  const textFieldDefaults = theme.components?.MuiTextField?.defaultProps ?? {};
  const computedVariant = other.variant ?? textFieldDefaults.variant ?? 'outlined';
  const computedSize = other.size ?? textFieldDefaults.size;
  return /*#__PURE__*/_jsx(MUITextField, _extends({
    variant: computedVariant,
    size: computedSize
  }, other, material, {
    inputProps: slotProps?.htmlInput,
    InputProps: transformInputProps(slotProps?.input),
    InputLabelProps: _extends({
      shrink: true
    }, slotProps?.inputLabel)
  }));
}
function BaseAutocomplete(props) {
  const rootProps = useGridRootProps();
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
    other = _objectWithoutPropertiesLoose(props, _excluded16);
  return /*#__PURE__*/_jsx(MUIAutocomplete, _extends({
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
        tagProps = _objectWithoutPropertiesLoose(_getTagProps, _excluded17);
      return /*#__PURE__*/_jsx(MUIChip, _extends({
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
        inputRest = _objectWithoutPropertiesLoose(params, _excluded18);
      return /*#__PURE__*/_jsx(MUITextField, _extends({}, inputRest, {
        label: label,
        placeholder: placeholder,
        inputProps: inputProps,
        InputProps: transformInputProps(InputProps, false),
        InputLabelProps: _extends({
          shrink: true
        }, InputLabelProps)
      }, slotProps?.textField, rootProps.slotProps?.baseTextField));
    }
  }, other, material));
}
function BaseInput(props) {
  return /*#__PURE__*/_jsx(MUIInputBase, _extends({}, transformInputProps(props)));
}
function transformInputProps(props, wrapAdornments = true) {
  if (!props) {
    return undefined;
  }
  const {
      slotProps,
      material
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded19);
  const result = other;
  if (wrapAdornments) {
    if (result.startAdornment) {
      result.startAdornment = /*#__PURE__*/_jsx(InputAdornment, {
        position: "start",
        children: result.startAdornment
      });
    }
    if (result.endAdornment) {
      result.endAdornment = /*#__PURE__*/_jsx(InputAdornment, {
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
      result.inputProps = _extends({}, result.inputProps, slotProps?.htmlInput);
    } else {
      result.inputProps = slotProps?.htmlInput;
    }
  }
  return result;
}
const BaseTextarea = forwardRef(function BaseTextarea(props, ref) {
  const {
      material
    } = props,
    other = _objectWithoutPropertiesLoose(props, _excluded20);
  return /*#__PURE__*/_jsx(MUITextareaAutosize, _extends({}, other, material, {
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
    other = _objectWithoutPropertiesLoose(props, _excluded21);
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
    content = p => wrappers(props, /*#__PURE__*/_jsx(MUIGrow, _extends({}, p.TransitionProps, {
      style: {
        transformOrigin: transformOrigin[p.placement]
      },
      onExited: handleExited(p.TransitionProps?.onExited),
      children: /*#__PURE__*/_jsx(MUIPaper, {
        children: children
      })
    })));
  }
  return /*#__PURE__*/_jsx(MUIPopper, _extends({
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
  return /*#__PURE__*/_jsx(ClickAwayListener, {
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
  return /*#__PURE__*/_jsx(MUIFocusTrap, {
    open: true,
    disableEnforceFocus: true,
    disableAutoFocus: true,
    children: /*#__PURE__*/_jsx("div", {
      tabIndex: -1,
      children: content
    })
  });
}
function BaseSelectOption(_ref) {
  let {
      native
    } = _ref,
    props = _objectWithoutPropertiesLoose(_ref, _excluded22);
  if (native) {
    return /*#__PURE__*/_jsx("option", _extends({}, props));
  }
  return /*#__PURE__*/_jsx(MUIMenuItem, _extends({}, props));
}
const StyledTabs = styled(MUITabs, {
  name: 'MuiDataGrid',
  slot: 'Tabs'
})(({
  theme
}) => ({
  borderBottom: `1px solid ${theme.palette.divider}`
}));
const StyledTab = styled(MUITab, {
  name: 'MuiDataGrid',
  slot: 'Tab'
})({
  flex: 1,
  minWidth: 'fit-content'
});
const StyledTabPanel = styled('div', {
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
    other = _objectWithoutPropertiesLoose(props, _excluded23);
  return /*#__PURE__*/_jsx(StyledTabPanel, _extends({
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
    props = _objectWithoutPropertiesLoose(_ref2, _excluded24);
  const id = useId();
  const labelId = `${id}-tab-${value}`;
  const panelId = `${id}-tabpanel-${value}`;
  return /*#__PURE__*/_jsxs(React.Fragment, {
    children: [/*#__PURE__*/_jsx(StyledTabs, _extends({}, props, {
      value: value,
      variant: "scrollable",
      scrollButtons: "auto"
    }, material, {
      children: items.map(item => /*#__PURE__*/_jsx(StyledTab, {
        value: item.value,
        label: item.label,
        id: labelId,
        "aria-controls": panelId
      }, item.value))
    })), items.map(item => /*#__PURE__*/_jsx(TabPanel, {
      value: item.value,
      active: value === item.value,
      id: panelId,
      "aria-labelledby": labelId,
      children: item.children
    }, item.value))]
  });
}
const iconSlots = {
  booleanCellTrueIcon: GridCheckIcon,
  booleanCellFalseIcon: GridCloseIcon,
  columnMenuIcon: GridTripleDotsVerticalIcon,
  openFilterButtonIcon: GridFilterListIcon,
  filterPanelDeleteIcon: GridCloseIcon,
  undoIcon: GridUndoIcon,
  redoIcon: GridRedoIcon,
  columnFilteredIcon: GridFilterAltIcon,
  columnSelectorIcon: GridColumnIcon,
  columnUnsortedIcon: GridColumnUnsortedIcon,
  columnSortedAscendingIcon: GridArrowUpwardIcon,
  columnSortedDescendingIcon: GridArrowDownwardIcon,
  columnResizeIcon: GridSeparatorIcon,
  densityCompactIcon: GridViewHeadlineIcon,
  densityStandardIcon: GridTableRowsIcon,
  densityComfortableIcon: GridViewStreamIcon,
  exportIcon: GridDownloadIcon,
  moreActionsIcon: GridMoreVertIcon,
  treeDataCollapseIcon: GridExpandMoreIcon,
  treeDataExpandIcon: GridKeyboardArrowRight,
  groupingCriteriaCollapseIcon: GridExpandMoreIcon,
  groupingCriteriaExpandIcon: GridKeyboardArrowRight,
  detailPanelExpandIcon: GridAddIcon,
  detailPanelCollapseIcon: GridRemoveIcon,
  rowReorderIcon: GridDragIcon,
  quickFilterIcon: GridSearchIcon,
  quickFilterClearIcon: GridClearIcon,
  columnMenuHideIcon: GridVisibilityOffIcon,
  columnMenuSortAscendingIcon: GridArrowUpwardIcon,
  columnMenuSortDescendingIcon: GridArrowDownwardIcon,
  columnMenuUnsortIcon: null,
  columnMenuFilterIcon: GridFilterAltIcon,
  columnMenuManageColumnsIcon: GridViewColumnIcon,
  columnMenuClearIcon: GridClearIcon,
  loadIcon: GridLoadIcon,
  filterPanelAddIcon: GridAddIcon,
  filterPanelRemoveAllIcon: GridDeleteForeverIcon,
  columnReorderIcon: GridDragIcon,
  menuItemCheckIcon: GridCheckIcon,
  longTextCellExpandIcon: GridLongTextCellExpandIcon,
  longTextCellCollapseIcon: GridLongTextCellCollapseIcon
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
const materialSlots = _extends({}, baseSlots, iconSlots);
export default materialSlots;