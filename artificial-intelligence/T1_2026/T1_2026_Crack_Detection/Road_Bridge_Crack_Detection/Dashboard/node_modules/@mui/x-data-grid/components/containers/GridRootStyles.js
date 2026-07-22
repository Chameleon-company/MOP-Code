"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.supportsColorMix = exports.colorMixIfSupported = exports.GridRootStyles = void 0;
var _styles = require("@mui/material/styles");
var _gridClasses = require("../../constants/gridClasses");
var _cssVariables = require("../../constants/cssVariables");
var _useGridSelector = require("../../hooks/utils/useGridSelector");
var _useGridPrivateApiContext = require("../../hooks/utils/useGridPrivateApiContext");
const columnSeparatorTargetSize = 10;
const columnSeparatorOffset = -5;
const focusOutlineWidth = 1;
const separatorIconDragStyles = {
  width: 3,
  rx: 1.5,
  x: 10.5
};

// Emotion thinks it knows better than us which selector we should use.
// https://github.com/emotion-js/emotion/issues/1105#issuecomment-1722524968
const ignoreSsrWarning = '/* emotion-disable-server-rendering-unsafe-selector-warning-please-do-not-use-this-the-warning-exists-for-a-reason */';
const shouldShowBorderTopRightRadiusSelector = apiRef => !apiRef.current.state.dimensions.isReady ? apiRef.current.state.dimensions.scrollbarSize === 0 : apiRef.current.state.dimensions.hasScrollX && (!apiRef.current.state.dimensions.hasScrollY || apiRef.current.state.dimensions.scrollbarSize === 0);
const GridRootStyles = exports.GridRootStyles = (0, _styles.styled)('div', {
  name: 'MuiDataGrid',
  slot: 'Root',
  overridesResolver: (props, styles) => {
    // Root overrides
    const overrides = [styles.root];
    _gridClasses.gridClassesOverrides.root.forEach(key => {
      overrides.push({
        [`&.${_gridClasses.gridClasses[key]}`]: styles[key]
      });
    });

    // Child element overrides
    // - Only declare overrides here for class names that are not applied to `styled` components.
    // - For `styled` components, declare overrides in the component itself.
    _gridClasses.gridClassesOverrides.children.forEach(key => {
      overrides.push({
        [`& .${_gridClasses.gridClasses[key]}`]: styles[key]
      });
    });
    return overrides;
  }
})(() => {
  const apiRef = (0, _useGridPrivateApiContext.useGridPrivateApiContext)();
  const shouldShowBorderTopRightRadius = (0, _useGridSelector.useGridSelector)(apiRef, shouldShowBorderTopRightRadiusSelector);
  const baseBackground = _cssVariables.vars.colors.background.base;
  const headerBackground = _cssVariables.vars.header.background.base;
  const pinnedBackground = _cssVariables.vars.cell.background.pinned;
  const hoverColor = removeOpacity(_cssVariables.vars.colors.interactive.hover);
  const hoverOpacity = _cssVariables.vars.colors.interactive.hoverOpacity;
  const selectedColor = _cssVariables.vars.colors.interactive.selected;
  const selectedOpacity = _cssVariables.vars.colors.interactive.selectedOpacity;
  const selectedHoverColor = selectedColor;
  const selectedHoverOpacity = `calc(${selectedOpacity} + ${hoverOpacity})`;
  const fallbackColors = {
    hover: _cssVariables.vars.colors.interactive.hover,
    selected: selectedColor,
    selectedHover: selectedColor
  };
  const hoverBackground = mix(baseBackground, hoverColor, hoverOpacity, fallbackColors.hover);
  const selectedBackground = mix(baseBackground, selectedColor, selectedOpacity, fallbackColors.selected);
  const selectedHoverBackground = mix(baseBackground, selectedHoverColor, selectedHoverOpacity, fallbackColors.selectedHover);
  const pinnedHoverBackground = mix(pinnedBackground, hoverColor, hoverOpacity, fallbackColors.hover);
  const pinnedSelectedBackground = mix(pinnedBackground, selectedColor, selectedOpacity, fallbackColors.selected);
  const pinnedSelectedHoverBackground = mix(pinnedBackground, selectedHoverColor, selectedHoverOpacity, fallbackColors.selectedHover);
  const getPinnedBackgroundStyles = backgroundColor => ({
    [`& .${_gridClasses.gridClasses['cell--pinnedLeft']}, & .${_gridClasses.gridClasses['cell--pinnedRight']}`]: {
      backgroundColor,
      '&.Mui-selected': {
        backgroundColor: mix(backgroundColor, selectedBackground, selectedOpacity, fallbackColors.selected),
        '&:hover': {
          backgroundColor: mix(backgroundColor, selectedHoverBackground, selectedHoverOpacity, fallbackColors.selectedHover)
        }
      }
    }
  });
  const pinnedHoverStyles = getPinnedBackgroundStyles(pinnedHoverBackground);
  const pinnedSelectedStyles = getPinnedBackgroundStyles(pinnedSelectedBackground);
  const pinnedSelectedHoverStyles = getPinnedBackgroundStyles(pinnedSelectedHoverBackground);
  const selectedStyles = {
    backgroundColor: selectedBackground,
    '&:hover': {
      backgroundColor: selectedHoverBackground,
      // Reset on touch devices, it doesn't add specificity
      '@media (hover: none)': {
        backgroundColor: selectedBackground
      }
    }
  };
  const gridStyle = {
    '--unstable_DataGrid-radius': _cssVariables.vars.radius.base,
    '--unstable_DataGrid-headWeight': _cssVariables.vars.typography.fontWeight.medium,
    '--DataGrid-rowBorderColor': _cssVariables.vars.colors.border.base,
    '--DataGrid-cellOffsetMultiplier': 2,
    '--DataGrid-width': '0px',
    '--DataGrid-hasScrollX': '0',
    '--DataGrid-hasScrollY': '0',
    '--DataGrid-scrollbarSize': '10px',
    '--DataGrid-rowWidth': '0px',
    '--DataGrid-columnsTotalWidth': '0px',
    '--DataGrid-leftPinnedWidth': '0px',
    '--DataGrid-rightPinnedWidth': '0px',
    '--DataGrid-headerHeight': '0px',
    '--DataGrid-headersTotalHeight': '0px',
    '--DataGrid-topContainerHeight': '0px',
    '--DataGrid-bottomContainerHeight': '0px',
    flex: 1,
    boxSizing: 'border-box',
    position: 'relative',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: _cssVariables.vars.colors.border.base,
    borderRadius: 'var(--unstable_DataGrid-radius)',
    backgroundColor: _cssVariables.vars.colors.background.base,
    color: _cssVariables.vars.colors.foreground.base,
    font: _cssVariables.vars.typography.font.body,
    outline: 'none',
    height: '100%',
    display: 'flex',
    minWidth: 0,
    // See https://github.com/mui/mui-x/issues/8547
    minHeight: 0,
    flexDirection: 'column',
    overflow: 'hidden',
    overflowAnchor: 'none',
    // Keep the same scrolling position
    transform: 'translate(0, 0)',
    // Create a stacking context to keep scrollbars from showing on top

    [`.${_gridClasses.gridClasses.main} > *:first-child${ignoreSsrWarning}`]: {
      borderTopLeftRadius: 'var(--unstable_DataGrid-radius)',
      borderTopRightRadius: 'var(--unstable_DataGrid-radius)'
    },
    [`&.${_gridClasses.gridClasses.autoHeight}`]: {
      height: 'auto'
    },
    [`&.${_gridClasses.gridClasses.autosizing}`]: {
      [`& .${_gridClasses.gridClasses.columnHeaderTitleContainerContent} > *`]: {
        overflow: 'visible !important'
      },
      '@media (hover: hover)': {
        [`& .${_gridClasses.gridClasses.menuIcon}`]: {
          width: '0 !important',
          visibility: 'hidden !important'
        }
      },
      [`& .${_gridClasses.gridClasses.cell}`]: {
        overflow: 'visible !important',
        whiteSpace: 'nowrap',
        minWidth: 'max-content !important',
        maxWidth: 'max-content !important'
      },
      [`& .${_gridClasses.gridClasses.groupingCriteriaCell}`]: {
        width: 'unset'
      },
      [`& .${_gridClasses.gridClasses.treeDataGroupingCell}`]: {
        width: 'unset'
      },
      [`& .${_gridClasses.gridClasses['columnHeader--filter']}`]: {
        flex: 'none !important',
        width: 'unset !important'
      }
    },
    [`&.${_gridClasses.gridClasses.withSidePanel}`]: {
      flexDirection: 'row'
    },
    [`& .${_gridClasses.gridClasses.mainContent}`]: {
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
      flex: 1
    },
    [`& .${_gridClasses.gridClasses.columnHeader}, & .${_gridClasses.gridClasses.cell}`]: {
      WebkitTapHighlightColor: 'transparent',
      padding: '0 10px',
      boxSizing: 'border-box'
    },
    [`& .${_gridClasses.gridClasses.columnHeader}:focus-within, & .${_gridClasses.gridClasses.cell}:focus-within`]: {
      outline: `solid ${setOpacity(_cssVariables.vars.colors.interactive.focus, 0.5)} ${focusOutlineWidth}px`,
      outlineOffset: focusOutlineWidth * -1
    },
    [`& .${_gridClasses.gridClasses.columnHeader}:focus, & .${_gridClasses.gridClasses.cell}:focus`]: {
      outline: `solid ${_cssVariables.vars.colors.interactive.focus} ${focusOutlineWidth}px`,
      outlineOffset: focusOutlineWidth * -1
    },
    // Hide the column separator when:
    // - the column is focused and has an outline
    // - the next column is focused and has an outline
    // - the column has a left or right border
    // - the next column is pinned right and has a left border
    [`& .${_gridClasses.gridClasses.columnHeader}:focus,
      & .${_gridClasses.gridClasses['columnHeader--withLeftBorder']},
      & .${_gridClasses.gridClasses['columnHeader--withRightBorder']},
      & .${_gridClasses.gridClasses['columnHeader--siblingFocused']}
      `]: {
      [`& .${_gridClasses.gridClasses.columnSeparator}`]: {
        opacity: 0
      },
      // Show resizable separators at all times on touch devices
      '@media (hover: none)': {
        [`& .${_gridClasses.gridClasses['columnSeparator--resizable']}`]: {
          opacity: 1
        }
      },
      [`& .${_gridClasses.gridClasses['columnSeparator--resizable']}:hover`]: {
        opacity: 1
      }
    },
    [`&.${_gridClasses.gridClasses['root--noToolbar']} [aria-rowindex="1"] [aria-colindex="1"]`]: {
      borderTopLeftRadius: 'calc(var(--unstable_DataGrid-radius) - 1px)'
    },
    [`&.${_gridClasses.gridClasses['root--noToolbar']} [aria-rowindex="1"] .${_gridClasses.gridClasses['columnHeader--last']}`]: {
      borderTopRightRadius: shouldShowBorderTopRightRadius ? 'calc(var(--unstable_DataGrid-radius) - 1px)' : undefined
    },
    [`& .${_gridClasses.gridClasses.columnHeaderCheckbox}, & .${_gridClasses.gridClasses.cellCheckbox}`]: {
      padding: 0,
      justifyContent: 'center',
      alignItems: 'center'
    },
    [`& .${_gridClasses.gridClasses.columnHeader}`]: {
      position: 'relative',
      display: 'flex',
      alignItems: 'center',
      backgroundColor: headerBackground
    },
    [`& .${_gridClasses.gridClasses.columnHeader} .${_gridClasses.gridClasses.sortButton}`]: {
      backgroundColor: _cssVariables.vars.header.background.base
    },
    [`& .${_gridClasses.gridClasses['columnHeader--filter']}`]: {
      paddingTop: 8,
      paddingBottom: 8,
      paddingRight: 5,
      minHeight: 'min-content',
      overflow: 'hidden'
    },
    [`&.${_gridClasses.gridClasses['root--densityCompact']} .${_gridClasses.gridClasses['columnHeader--filter']}`]: {
      paddingTop: 4,
      paddingBottom: 4
    },
    [`& .${_gridClasses.gridClasses['virtualScroller--hasScrollX']} .${_gridClasses.gridClasses['columnHeader--last']}`]: {
      overflow: 'hidden'
    },
    [`& .${_gridClasses.gridClasses['pivotPanelField--sorted']} .${_gridClasses.gridClasses.iconButtonContainer},
      & .${_gridClasses.gridClasses['columnHeader--sorted']} .${_gridClasses.gridClasses.iconButtonContainer},
      & .${_gridClasses.gridClasses['columnHeader--filtered']} .${_gridClasses.gridClasses.iconButtonContainer}`]: {
      visibility: 'visible',
      width: 'auto'
    },
    [`& .${_gridClasses.gridClasses.pivotPanelField}:not(.${_gridClasses.gridClasses['pivotPanelField--sorted']}) .${_gridClasses.gridClasses.sortButton},
      & .${_gridClasses.gridClasses.columnHeader}:not(.${_gridClasses.gridClasses['columnHeader--sorted']}) .${_gridClasses.gridClasses.sortButton}`]: {
      opacity: 0,
      transition: _cssVariables.vars.transition(['opacity'], {
        duration: _cssVariables.vars.transitions.duration.short
      })
    },
    [`& .${_gridClasses.gridClasses.columnHeaderTitleContainer}`]: {
      display: 'flex',
      alignItems: 'center',
      gap: _cssVariables.vars.spacing(0.25),
      minWidth: 0,
      flex: 1,
      whiteSpace: 'nowrap',
      overflow: 'hidden'
    },
    [`& .${_gridClasses.gridClasses.columnHeaderTitleContainerContent}`]: {
      overflow: 'hidden',
      display: 'flex',
      alignItems: 'center'
    },
    [`& .${_gridClasses.gridClasses['columnHeader--filledGroup']} .${_gridClasses.gridClasses.columnHeaderTitleContainer}`]: {
      borderBottomWidth: '1px',
      borderBottomStyle: 'solid',
      boxSizing: 'border-box'
    },
    [`& .${_gridClasses.gridClasses.sortIcon}, & .${_gridClasses.gridClasses.filterIcon}`]: {
      fontSize: 'inherit'
    },
    [`& .${_gridClasses.gridClasses['columnHeader--sortable']}`]: {
      cursor: 'pointer'
    },
    [`& .${_gridClasses.gridClasses['columnHeader--alignCenter']} .${_gridClasses.gridClasses.columnHeaderTitleContainer}`]: {
      justifyContent: 'center'
    },
    [`& .${_gridClasses.gridClasses['columnHeader--alignRight']} .${_gridClasses.gridClasses.columnHeaderDraggableContainer}, & .${_gridClasses.gridClasses['columnHeader--alignRight']} .${_gridClasses.gridClasses.columnHeaderTitleContainer}`]: {
      flexDirection: 'row-reverse'
    },
    [`& .${_gridClasses.gridClasses['columnHeader--alignCenter']} .${_gridClasses.gridClasses.menuIcon}`]: {
      marginLeft: 'auto'
    },
    [`& .${_gridClasses.gridClasses['columnHeader--alignRight']} .${_gridClasses.gridClasses.menuIcon}`]: {
      marginRight: 'auto',
      marginLeft: -5
    },
    [`& .${_gridClasses.gridClasses['columnHeader--moving']}`]: {
      backgroundColor: hoverBackground
    },
    [`& .${_gridClasses.gridClasses['columnHeader--pinnedLeft']}, & .${_gridClasses.gridClasses['columnHeader--pinnedRight']}`]: {
      position: 'sticky',
      zIndex: 40,
      // Should be above the column separator
      background: _cssVariables.vars.header.background.base
    },
    [`& .${_gridClasses.gridClasses.columnSeparator}`]: {
      position: 'absolute',
      overflow: 'hidden',
      zIndex: 30,
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      maxWidth: columnSeparatorTargetSize,
      color: _cssVariables.vars.colors.border.base
    },
    [`& .${_gridClasses.gridClasses.columnHeaders}`]: {
      width: 'var(--DataGrid-rowWidth)',
      backgroundColor: headerBackground
    },
    '@media (hover: hover)': {
      [`& .${_gridClasses.gridClasses.columnHeader}:hover`]: {
        [`& .${_gridClasses.gridClasses.menuIcon}`]: {
          width: 'auto',
          visibility: 'visible'
        },
        [`& .${_gridClasses.gridClasses.iconButtonContainer}`]: {
          visibility: 'visible',
          width: 'auto'
        }
      },
      [`& .${_gridClasses.gridClasses.columnHeader}:not(.${_gridClasses.gridClasses['columnHeader--sorted']}):hover .${_gridClasses.gridClasses.sortButton},
        & .${_gridClasses.gridClasses.pivotPanelField}:not(.${_gridClasses.gridClasses['pivotPanelField--sorted']}):hover .${_gridClasses.gridClasses.sortButton},
        & .${_gridClasses.gridClasses.pivotPanelField}:not(.${_gridClasses.gridClasses['pivotPanelField--sorted']}) .${_gridClasses.gridClasses.sortButton}:focus-visible`]: {
        opacity: 1
      },
      // Add opacity only to the button content to avoid affecting the background color
      [`& .${_gridClasses.gridClasses.columnHeader}:not(.${_gridClasses.gridClasses['columnHeader--sorted']}):hover .${_gridClasses.gridClasses.sortButton} > *,
        & .${_gridClasses.gridClasses.pivotPanelField}:not(.${_gridClasses.gridClasses['pivotPanelField--sorted']}):hover .${_gridClasses.gridClasses.sortButton} > *,
        & .${_gridClasses.gridClasses.pivotPanelField}:not(.${_gridClasses.gridClasses['pivotPanelField--sorted']}) .${_gridClasses.gridClasses.sortButton}:focus-visible > *`]: {
        opacity: 0.78
      },
      [`& .${_gridClasses.gridClasses.pivotPanelFieldActionContainer} button:hover`]: {
        backgroundColor: _cssVariables.vars.colors.background.base
      }
    },
    '@media (hover: none)': {
      [`& .${_gridClasses.gridClasses.columnHeader} .${_gridClasses.gridClasses.menuIcon}`]: {
        width: 'auto',
        visibility: 'visible'
      },
      [`& .${_gridClasses.gridClasses.columnHeader}:focus,
        & .${_gridClasses.gridClasses['columnHeader--siblingFocused']}`]: {
        [`.${_gridClasses.gridClasses['columnSeparator--resizable']}`]: {
          color: _cssVariables.vars.colors.foreground.accent
        }
      },
      [`& .${_gridClasses.gridClasses.pivotPanelField}:not(.${_gridClasses.gridClasses['pivotPanelField--sorted']}) .${_gridClasses.gridClasses.sortButton}`]: {
        opacity: 0.78
      }
    },
    // Hide the column separator when the column has border and it is not resizable
    // In this case, this column separator may block interaction with the separator from the adjacent column that is resizable
    [`& .${_gridClasses.gridClasses['columnHeader--withLeftBorder']} .${_gridClasses.gridClasses['columnSeparator--sideLeft']}:not(.${_gridClasses.gridClasses['columnSeparator--resizable']}), & .${_gridClasses.gridClasses['columnHeader--withRightBorder']} .${_gridClasses.gridClasses['columnSeparator--sideRight']}:not(.${_gridClasses.gridClasses['columnSeparator--resizable']})`]: {
      display: 'none'
    },
    [`& .${_gridClasses.gridClasses['columnSeparator--sideLeft']}`]: {
      left: columnSeparatorOffset
    },
    [`& .${_gridClasses.gridClasses['columnSeparator--sideRight']}`]: {
      right: columnSeparatorOffset
    },
    [`& .${_gridClasses.gridClasses['columnHeader--withRightBorder']} .${_gridClasses.gridClasses['columnSeparator--sideLeft']}`]: {
      left: columnSeparatorOffset - 0.5
    },
    [`& .${_gridClasses.gridClasses['columnHeader--withRightBorder']} .${_gridClasses.gridClasses['columnSeparator--sideRight']}`]: {
      right: columnSeparatorOffset - 0.5
    },
    [`& .${_gridClasses.gridClasses['columnSeparator--resizable']}`]: {
      cursor: 'col-resize',
      touchAction: 'none',
      [`&.${_gridClasses.gridClasses['columnSeparator--resizing']}`]: {
        color: _cssVariables.vars.colors.foreground.accent
      },
      // Always appear as draggable on touch devices
      '@media (hover: none)': {
        [`& .${_gridClasses.gridClasses.iconSeparator} rect`]: separatorIconDragStyles
      },
      '@media (hover: hover)': {
        '&:hover': {
          color: _cssVariables.vars.colors.foreground.accent,
          [`& .${_gridClasses.gridClasses.iconSeparator} rect`]: separatorIconDragStyles
        }
      },
      '& svg': {
        pointerEvents: 'none'
      }
    },
    [`& .${_gridClasses.gridClasses.iconSeparator}`]: {
      color: 'inherit',
      transition: _cssVariables.vars.transition(['color', 'width'], {
        duration: _cssVariables.vars.transitions.duration.short
      })
    },
    [`& .${_gridClasses.gridClasses.menuIcon}`]: {
      width: 0,
      visibility: 'hidden',
      fontSize: 20,
      marginRight: -5,
      display: 'flex',
      alignItems: 'center'
    },
    [`.${_gridClasses.gridClasses.menuOpen}`]: {
      visibility: 'visible',
      width: 'auto'
    },
    [`& .${_gridClasses.gridClasses.headerFilterRow}`]: {
      [`& .${_gridClasses.gridClasses.columnHeader}, & .${_gridClasses.gridClasses.scrollbarFiller}`]: {
        boxSizing: 'border-box',
        borderBottom: '1px solid var(--DataGrid-rowBorderColor)'
      }
    },
    /* Bottom border of the top-container */
    [`& .${_gridClasses.gridClasses['row--borderBottom']} .${_gridClasses.gridClasses.columnHeader},
      & .${_gridClasses.gridClasses['row--borderBottom']} .${_gridClasses.gridClasses.filler},
      & .${_gridClasses.gridClasses['row--borderBottom']} .${_gridClasses.gridClasses.scrollbarFiller}`]: {
      borderBottom: `1px solid var(--DataGrid-rowBorderColor)`
    },
    [`& .${_gridClasses.gridClasses['row--borderBottom']} .${_gridClasses.gridClasses.cell}`]: {
      borderBottom: `1px solid var(--rowBorderColor)`
    },
    /* Row styles */
    [`.${_gridClasses.gridClasses.row}`]: {
      display: 'flex',
      width: 'var(--DataGrid-rowWidth)',
      breakInside: 'avoid',
      // Avoid the row to be broken in two different print pages.

      '--rowBorderColor': 'var(--DataGrid-rowBorderColor)',
      [`&.${_gridClasses.gridClasses['row--firstVisible']}`]: {
        '--rowBorderColor': 'transparent'
      },
      '&:hover': {
        backgroundColor: hoverBackground,
        // Reset on touch devices, it doesn't add specificity
        '@media (hover: none)': {
          backgroundColor: 'transparent'
        }
      },
      [`&.${_gridClasses.gridClasses.rowSkeleton}:hover`]: {
        backgroundColor: 'transparent'
      },
      '&.Mui-selected': selectedStyles,
      position: 'relative'
    },
    /* Cell styles */
    [`& .${_gridClasses.gridClasses.cell}`]: {
      flex: '0 0 auto',
      height: 'var(--height)',
      width: 'var(--width)',
      lineHeight: 'calc(var(--height) - 1px)',
      // -1px for the border

      boxSizing: 'border-box',
      borderTop: `1px solid var(--rowBorderColor)`,
      overflow: 'hidden',
      whiteSpace: 'nowrap',
      textOverflow: 'ellipsis',
      '&.Mui-selected': selectedStyles
    },
    [`& .${_gridClasses.gridClasses['virtualScrollerContent--overflowed']} .${_gridClasses.gridClasses['row--lastVisible']} .${_gridClasses.gridClasses.cell}`]: {
      borderTopColor: 'transparent'
    },
    [`& .${_gridClasses.gridClasses.pinnedRows} .${_gridClasses.gridClasses.row}, .${_gridClasses.gridClasses.aggregationRowOverlayWrapper} .${_gridClasses.gridClasses.row}`]: {
      backgroundColor: pinnedBackground,
      '&:hover': {
        backgroundColor: pinnedHoverBackground
      }
    },
    [`& .${_gridClasses.gridClasses['pinnedRows--top']} :first-of-type`]: {
      [`& .${_gridClasses.gridClasses.cell}, .${_gridClasses.gridClasses.scrollbarFiller}`]: {
        borderTop: 'none'
      }
    },
    [`&.${_gridClasses.gridClasses['root--disableUserSelection']}`]: {
      userSelect: 'none'
    },
    [`& .${_gridClasses.gridClasses['row--dynamicHeight']} > .${_gridClasses.gridClasses.cell}`]: {
      whiteSpace: 'initial',
      lineHeight: 'inherit'
    },
    [`& .${_gridClasses.gridClasses.cellEmpty}`]: {
      flex: 1,
      padding: 0,
      height: 'unset'
    },
    [`& .${_gridClasses.gridClasses.cell}.${_gridClasses.gridClasses['cell--selectionMode']}`]: {
      cursor: 'default'
    },
    [`& .${_gridClasses.gridClasses.cell}.${_gridClasses.gridClasses['cell--editing']}`]: {
      padding: 1,
      display: 'flex',
      boxShadow: _cssVariables.vars.shadows.base,
      backgroundColor: _cssVariables.vars.colors.background.overlay,
      '&:focus-within': {
        outline: `${focusOutlineWidth}px solid ${_cssVariables.vars.colors.interactive.focus}`,
        outlineOffset: focusOutlineWidth * -1
      }
    },
    [`& .${_gridClasses.gridClasses['cell--editing']}`]: {
      '& .MuiInputBase-root': {
        height: '100%'
      }
    },
    [`& .${_gridClasses.gridClasses['row--editing']}`]: {
      boxShadow: _cssVariables.vars.shadows.base
    },
    [`& .${_gridClasses.gridClasses['row--editing']} .${_gridClasses.gridClasses.cell}`]: {
      boxShadow: 'none',
      backgroundColor: _cssVariables.vars.colors.background.overlay
    },
    [`& .${_gridClasses.gridClasses.editBooleanCell}`]: {
      display: 'flex',
      height: '100%',
      width: '100%',
      alignItems: 'center',
      justifyContent: 'center'
    },
    [`& .${_gridClasses.gridClasses.booleanCell}[data-value="true"]`]: {
      color: _cssVariables.vars.colors.foreground.muted
    },
    [`& .${_gridClasses.gridClasses.booleanCell}[data-value="false"]`]: {
      color: _cssVariables.vars.colors.foreground.disabled
    },
    [`& .${_gridClasses.gridClasses.actionsCell}`]: {
      display: 'inline-flex',
      alignItems: 'center',
      gridGap: _cssVariables.vars.spacing(1)
    },
    [`& .${_gridClasses.gridClasses.rowReorderCell}`]: {
      display: 'inline-flex',
      flex: 1,
      alignItems: 'center',
      justifyContent: 'center',
      opacity: _cssVariables.vars.colors.interactive.disabledOpacity
    },
    [`& .${_gridClasses.gridClasses['rowReorderCell--draggable']}`]: {
      cursor: 'grab',
      opacity: 1
    },
    [`& .${_gridClasses.gridClasses.rowReorderCellContainer}`]: {
      padding: 0,
      display: 'flex',
      alignItems: 'stretch'
    },
    [`.${_gridClasses.gridClasses.withBorderColor}`]: {
      borderColor: _cssVariables.vars.colors.border.base
    },
    [`& .${_gridClasses.gridClasses['cell--withLeftBorder']}, & .${_gridClasses.gridClasses['columnHeader--withLeftBorder']}`]: {
      borderLeftColor: 'var(--DataGrid-rowBorderColor)',
      borderLeftWidth: '1px',
      borderLeftStyle: 'solid'
    },
    [`& .${_gridClasses.gridClasses['cell--withRightBorder']}, & .${_gridClasses.gridClasses['columnHeader--withRightBorder']}`]: {
      borderRightColor: 'var(--DataGrid-rowBorderColor)',
      borderRightWidth: '1px',
      borderRightStyle: 'solid'
    },
    [`& .${_gridClasses.gridClasses['cell--flex']}`]: {
      display: 'flex',
      alignItems: 'center',
      lineHeight: 'inherit'
    },
    [`& .${_gridClasses.gridClasses['cell--textLeft']}`]: {
      textAlign: 'left',
      justifyContent: 'flex-start'
    },
    [`& .${_gridClasses.gridClasses['cell--textRight']}`]: {
      textAlign: 'right',
      justifyContent: 'flex-end'
    },
    [`& .${_gridClasses.gridClasses['cell--textCenter']}`]: {
      textAlign: 'center',
      justifyContent: 'center'
    },
    [`& .${_gridClasses.gridClasses['cell--pinnedLeft']}, & .${_gridClasses.gridClasses['cell--pinnedRight']}`]: {
      position: 'sticky',
      zIndex: 30,
      background: _cssVariables.vars.cell.background.pinned,
      '&.Mui-selected': {
        backgroundColor: pinnedSelectedBackground
      }
    },
    [`& .${_gridClasses.gridClasses.row}`]: {
      '&:hover': pinnedHoverStyles,
      '&.Mui-selected': pinnedSelectedStyles,
      '&.Mui-selected:hover': pinnedSelectedHoverStyles
    },
    [`& .${_gridClasses.gridClasses.cellOffsetLeft}`]: {
      flex: '0 0 auto',
      display: 'inline-block'
    },
    [`& .${_gridClasses.gridClasses.cellSkeleton}`]: {
      flex: '0 0 auto',
      height: '100%',
      display: 'inline-flex',
      alignItems: 'center'
    },
    [`& .${_gridClasses.gridClasses.columnHeaderDraggableContainer}`]: {
      display: 'flex',
      width: '100%',
      height: '100%'
    },
    [`& .${_gridClasses.gridClasses.rowReorderCellPlaceholder}`]: {
      display: 'none'
    },
    [`& .${_gridClasses.gridClasses['columnHeader--dragging']}`]: {
      background: _cssVariables.vars.colors.background.overlay,
      padding: '0 12px',
      borderRadius: 'var(--unstable_DataGrid-radius)',
      opacity: _cssVariables.vars.colors.interactive.disabledOpacity
    },
    [`& .${_gridClasses.gridClasses['row--dragging']}`]: {
      background: _cssVariables.vars.colors.background.overlay,
      padding: '0 12px',
      borderRadius: 'var(--unstable_DataGrid-radius)',
      border: '1px solid var(--DataGrid-rowBorderColor)',
      color: _cssVariables.vars.colors.foreground.base,
      transform: 'translateZ(0)',
      [`& .${_gridClasses.gridClasses.rowReorderCellPlaceholder}`]: {
        padding: '0 6px',
        display: 'flex'
      }
    },
    [`& .${_gridClasses.gridClasses.treeDataGroupingCell}`]: {
      display: 'flex',
      alignItems: 'center',
      width: '100%'
    },
    [`& .${_gridClasses.gridClasses.treeDataGroupingCellToggle}`]: {
      flex: '0 0 28px',
      alignSelf: 'stretch',
      marginRight: _cssVariables.vars.spacing(2)
    },
    [`& .${_gridClasses.gridClasses.treeDataGroupingCellLoadingContainer}, .${_gridClasses.gridClasses.groupingCriteriaCellLoadingContainer}`]: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%'
    },
    [`& .${_gridClasses.gridClasses.groupingCriteriaCell}`]: {
      display: 'flex',
      alignItems: 'center',
      width: '100%'
    },
    [`& .${_gridClasses.gridClasses.groupingCriteriaCellToggle}`]: {
      flex: '0 0 28px',
      alignSelf: 'stretch',
      marginRight: _cssVariables.vars.spacing(2)
    },
    /* ScrollbarFiller styles */
    [`& .${_gridClasses.gridClasses.columnHeaders} .${_gridClasses.gridClasses.scrollbarFiller}`]: {
      backgroundColor: headerBackground
    },
    [`.${_gridClasses.gridClasses.scrollbarFiller}`]: {
      minWidth: 'calc(var(--DataGrid-hasScrollY) * var(--DataGrid-scrollbarSize))',
      alignSelf: 'stretch',
      backgroundColor: headerBackground,
      [`&.${_gridClasses.gridClasses['scrollbarFiller--pinnedRight']}`]: {
        position: 'sticky',
        zIndex: 40,
        // Should be above the column separator
        right: 0
      }
    },
    [`& .${_gridClasses.gridClasses.filler}`]: {
      flex: '1 0 auto'
    },
    [`& .${_gridClasses.gridClasses['filler--borderBottom']}`]: {
      borderBottom: '1px solid var(--DataGrid-rowBorderColor)'
    },
    [`& .${_gridClasses.gridClasses.columnHeaders} .${_gridClasses.gridClasses.filler}`]: {
      backgroundColor: headerBackground
    },
    /* Used when skeleton/no columns overlay is visible */
    [`& .${_gridClasses.gridClasses['main--hiddenContent']}`]: {
      [`& .${_gridClasses.gridClasses.virtualScrollerContent}`]: {
        // We use visibility hidden so that the virtual scroller content retains its height.
        // Position fixed is used to remove the virtual scroller content from the flow.
        // https://github.com/mui/mui-x/issues/14061
        position: 'fixed',
        visibility: 'hidden'
      },
      // Hide grid content
      [`& .${_gridClasses.gridClasses.pinnedRows}`]: {
        display: 'none'
      }
    },
    [`& .${_gridClasses.gridClasses['row--beingDragged']}`]: {
      color: _cssVariables.vars.colors.foreground.disabled,
      '&:hover': {
        backgroundColor: 'transparent'
      }
    }
  };
  return gridStyle;
});
function setOpacity(color, opacity) {
  return `rgba(from ${color} r g b / ${opacity})`;
}
function removeOpacity(color) {
  return setOpacity(color, 1);
}
const supportsColorMix = exports.supportsColorMix = typeof CSS !== 'undefined' && typeof CSS.supports === 'function' && CSS.supports('color', 'color-mix(in srgb, red 50%, blue 50%)');
const colorMixIfSupported = (colorMixValue, fallback) => {
  if (!supportsColorMix) {
    return fallback;
  }
  return colorMixValue;
};
exports.colorMixIfSupported = colorMixIfSupported;
function mix(background, overlay, opacity, fallback) {
  return colorMixIfSupported(`color-mix(in srgb,${background}, ${overlay} calc(${opacity} * 100%))`, fallback);
}