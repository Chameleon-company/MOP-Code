"use strict";
'use client';

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.GridColumnHeaderItem = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var React = _interopRequireWildcard(require("react"));
var _propTypes = _interopRequireDefault(require("prop-types"));
var _clsx = _interopRequireDefault(require("clsx"));
var _composeClasses = _interopRequireDefault(require("@mui/utils/composeClasses"));
var _capitalize = _interopRequireDefault(require("@mui/utils/capitalize"));
var _useId = _interopRequireDefault(require("@mui/utils/useId"));
var _fastMemo = require("@mui/x-internals/fastMemo");
var _RtlProvider = require("@mui/system/RtlProvider");
var _doesSupportPreventScroll = require("../../utils/doesSupportPreventScroll");
var _useGridPrivateApiContext = require("../../hooks/utils/useGridPrivateApiContext");
var _getColumnMenuItemKeys = require("../../hooks/features/columnMenu/getColumnMenuItemKeys");
var _ColumnHeaderMenuIcon = require("./ColumnHeaderMenuIcon");
var _GridColumnHeaderMenu = require("../menu/columnMenu/GridColumnHeaderMenu");
var _gridClasses = require("../../constants/gridClasses");
var _useGridRootProps = require("../../hooks/utils/useGridRootProps");
var _GridGenericColumnHeaderItem = require("./GridGenericColumnHeaderItem");
var _domUtils = require("../../utils/domUtils");
var _constants = require("../../internals/constants");
var _utils = require("../../internals/utils");
var _jsxRuntime = require("react/jsx-runtime");
const useUtilityClasses = ownerState => {
  const {
    disableColumnSorting
  } = (0, _useGridRootProps.useGridRootProps)();
  const {
    colDef,
    classes,
    isDragging,
    sortDirection,
    showRightBorder,
    showLeftBorder,
    filterItemsCounter,
    pinnedPosition,
    isSiblingFocused
  } = ownerState;
  const isColumnSortable = colDef.sortable && !disableColumnSorting;
  const isColumnSorted = sortDirection != null;
  const isColumnFiltered = filterItemsCounter != null && filterItemsCounter > 0;
  // todo refactor to a prop on col isNumeric or ?? ie: coltype===price wont work
  const isColumnNumeric = colDef.type === 'number';
  const slots = {
    root: ['columnHeader', colDef.headerAlign && `columnHeader--align${(0, _capitalize.default)(colDef.headerAlign)}`, isColumnSortable && 'columnHeader--sortable', isDragging && 'columnHeader--moving', isColumnSorted && 'columnHeader--sorted', isColumnFiltered && 'columnHeader--filtered', isColumnNumeric && 'columnHeader--numeric', 'withBorderColor', showRightBorder && 'columnHeader--withRightBorder', showLeftBorder && 'columnHeader--withLeftBorder', pinnedPosition === _constants.PinnedColumnPosition.LEFT && 'columnHeader--pinnedLeft', pinnedPosition === _constants.PinnedColumnPosition.RIGHT && 'columnHeader--pinnedRight',
    // TODO: Remove classes below and restore `:has` selectors when they are supported in jsdom
    // See https://github.com/mui/mui-x/pull/14559
    isSiblingFocused && 'columnHeader--siblingFocused'],
    draggableContainer: ['columnHeaderDraggableContainer'],
    titleContainer: ['columnHeaderTitleContainer'],
    titleContainerContent: ['columnHeaderTitleContainerContent']
  };
  return (0, _composeClasses.default)(slots, _gridClasses.getDataGridUtilityClass, classes);
};
function GridColumnHeaderItem(props) {
  const {
    colDef,
    columnMenuOpen,
    colIndex,
    headerHeight,
    isResizing,
    isLast,
    sortDirection,
    sortIndex,
    filterItemsCounter,
    hasFocus,
    tabIndex,
    disableReorder,
    separatorSide,
    showLeftBorder,
    showRightBorder,
    pinnedPosition,
    pinnedOffset
  } = props;
  const apiRef = (0, _useGridPrivateApiContext.useGridPrivateApiContext)();
  const rootProps = (0, _useGridRootProps.useGridRootProps)();
  const isRtl = (0, _RtlProvider.useRtl)();
  const headerCellRef = React.useRef(null);
  const columnMenuId = (0, _useId.default)();
  const columnMenuButtonId = (0, _useId.default)();
  const iconButtonRef = React.useRef(null);
  const [showColumnMenuIcon, setShowColumnMenuIcon] = React.useState(columnMenuOpen);
  const columnMenuSlotProps = rootProps.slotProps?.columnMenu;
  const columnMenuComponent = rootProps.slots.columnMenu;
  const defaultSlots = columnMenuComponent?.defaultSlots;
  const defaultSlotProps = columnMenuComponent?.defaultSlotProps;
  const hasKnownDefaultColumnMenu = defaultSlots != null && defaultSlotProps != null;
  const columnMenuItemKeys = React.useMemo(() => {
    if (!hasKnownDefaultColumnMenu) {
      return [];
    }
    return (0, _getColumnMenuItemKeys.getColumnMenuItemKeys)({
      apiRef,
      colDef,
      defaultSlots,
      defaultSlotProps,
      slots: columnMenuSlotProps?.slots,
      slotProps: columnMenuSlotProps?.slotProps
    });
  }, [apiRef, colDef, defaultSlotProps, defaultSlots, hasKnownDefaultColumnMenu, columnMenuSlotProps?.slotProps, columnMenuSlotProps?.slots]);

  // If we don't have a "known" default column menu (i.e. a custom menu component
  // without `defaultSlots` / `defaultSlotProps` statics), we treat it as opaque
  // and assume it has items, so we always show the column menu icon.
  // Only the built-in/default menu path can hide the icon when there are no items.
  const hasColumnMenuItems = !hasKnownDefaultColumnMenu || columnMenuItemKeys.length > 0;
  const isDraggable = !rootProps.disableColumnReorder && !disableReorder && !colDef.disableReorder;
  let headerComponent;
  if (colDef.renderHeader) {
    headerComponent = colDef.renderHeader(apiRef.current.getColumnHeaderParams(colDef.field));
  }
  const ownerState = (0, _extends2.default)({}, props, {
    classes: rootProps.classes,
    showRightBorder,
    showLeftBorder
  });
  const classes = useUtilityClasses(ownerState);
  const publish = React.useCallback(eventName => event => {
    // Ignore portal
    // See https://github.com/mui/mui-x/issues/1721
    if ((0, _domUtils.isEventTargetInPortal)(event)) {
      return;
    }
    apiRef.current.publishEvent(eventName, apiRef.current.getColumnHeaderParams(colDef.field), event);
  }, [apiRef, colDef.field]);
  const mouseEventsHandlers = React.useMemo(() => ({
    onClick: publish('columnHeaderClick'),
    onContextMenu: publish('columnHeaderContextMenu'),
    onDoubleClick: publish('columnHeaderDoubleClick'),
    onMouseOver: publish('columnHeaderOver'),
    // TODO remove as it's not used
    onMouseOut: publish('columnHeaderOut'),
    // TODO remove as it's not used
    onMouseEnter: publish('columnHeaderEnter'),
    // TODO remove as it's not used
    onMouseLeave: publish('columnHeaderLeave'),
    // TODO remove as it's not used
    onKeyDown: publish('columnHeaderKeyDown'),
    onFocus: publish('columnHeaderFocus'),
    onBlur: publish('columnHeaderBlur')
  }), [publish]);
  const draggableEventHandlers = React.useMemo(() => isDraggable ? {
    onDragStart: publish('columnHeaderDragStart'),
    onDragEnter: publish('columnHeaderDragEnter'),
    onDragOver: publish('columnHeaderDragOver'),
    onDragEndCapture: publish('columnHeaderDragEnd')
  } : {}, [isDraggable, publish]);
  const columnHeaderSeparatorProps = React.useMemo(() => ({
    onMouseDown: publish('columnSeparatorMouseDown'),
    onDoubleClick: publish('columnSeparatorDoubleClick')
  }), [publish]);
  React.useEffect(() => {
    if (!showColumnMenuIcon && columnMenuOpen) {
      setShowColumnMenuIcon(columnMenuOpen);
    }
  }, [showColumnMenuIcon, columnMenuOpen]);
  React.useEffect(() => {
    if (hasKnownDefaultColumnMenu && columnMenuOpen && !hasColumnMenuItems) {
      apiRef.current.hideColumnMenu();
    }
  }, [apiRef, columnMenuOpen, hasColumnMenuItems, hasKnownDefaultColumnMenu]);
  const handleExited = React.useCallback(() => {
    setShowColumnMenuIcon(false);
  }, []);
  const columnMenuIconButton = !rootProps.disableColumnMenu && !colDef.disableColumnMenu && hasColumnMenuItems && /*#__PURE__*/(0, _jsxRuntime.jsx)(_ColumnHeaderMenuIcon.ColumnHeaderMenuIcon, {
    colDef: colDef,
    columnMenuId: columnMenuId,
    columnMenuButtonId: columnMenuButtonId,
    open: showColumnMenuIcon,
    iconButtonRef: iconButtonRef
  });
  const columnMenu = /*#__PURE__*/(0, _jsxRuntime.jsx)(_GridColumnHeaderMenu.GridColumnHeaderMenu, {
    columnMenuId: columnMenuId,
    columnMenuButtonId: columnMenuButtonId,
    field: colDef.field,
    open: columnMenuOpen,
    target: iconButtonRef.current,
    ContentComponent: rootProps.slots.columnMenu,
    contentComponentProps: rootProps.slotProps?.columnMenu,
    onExited: handleExited
  });
  const sortingOrder = colDef.sortingOrder ?? rootProps.sortingOrder;
  const showSortIcon = (colDef.sortable || sortDirection != null) && !colDef.hideSortIcons && !rootProps.disableColumnSorting;
  const columnTitleIconButtons = /*#__PURE__*/(0, _jsxRuntime.jsxs)(React.Fragment, {
    children: [!rootProps.disableColumnFilter && /*#__PURE__*/(0, _jsxRuntime.jsx)(rootProps.slots.columnHeaderFilterIconButton, (0, _extends2.default)({
      field: colDef.field,
      counter: filterItemsCounter
    }, rootProps.slotProps?.columnHeaderFilterIconButton)), showSortIcon && /*#__PURE__*/(0, _jsxRuntime.jsx)(rootProps.slots.columnHeaderSortIcon, (0, _extends2.default)({
      field: colDef.field,
      direction: sortDirection,
      index: sortIndex,
      sortingOrder: sortingOrder,
      disabled: !colDef.sortable
    }, rootProps.slotProps?.columnHeaderSortIcon))]
  });
  React.useLayoutEffect(() => {
    const columnMenuState = apiRef.current.state.columnMenu;
    if (hasFocus && !columnMenuState.open) {
      const focusableElement = headerCellRef.current.querySelector('[tabindex="0"]');
      const elementToFocus = focusableElement || headerCellRef.current;
      if (!elementToFocus) {
        return;
      }
      if ((0, _doesSupportPreventScroll.doesSupportPreventScroll)()) {
        elementToFocus.focus({
          preventScroll: true
        });
      } else {
        const scrollPosition = apiRef.current.getScrollPosition();
        elementToFocus.focus();
        apiRef.current.scroll(scrollPosition);
      }
    }
  }, [apiRef, hasFocus]);
  const headerClassName = typeof colDef.headerClassName === 'function' ? colDef.headerClassName({
    field: colDef.field,
    colDef
  }) : colDef.headerClassName;
  const label = colDef.headerName ?? colDef.field;
  const style = React.useMemo(() => (0, _utils.attachPinnedStyle)((0, _extends2.default)({}, props.style), isRtl, pinnedPosition, pinnedOffset), [pinnedPosition, pinnedOffset, props.style, isRtl]);
  return /*#__PURE__*/(0, _jsxRuntime.jsx)(_GridGenericColumnHeaderItem.GridGenericColumnHeaderItem, (0, _extends2.default)({
    ref: headerCellRef,
    classes: classes,
    columnMenuOpen: columnMenuOpen,
    colIndex: colIndex,
    height: headerHeight,
    isResizing: isResizing,
    sortDirection: sortDirection,
    hasFocus: hasFocus,
    tabIndex: tabIndex,
    separatorSide: separatorSide,
    isDraggable: isDraggable,
    headerComponent: headerComponent,
    description: colDef.description,
    elementId: colDef.field,
    width: colDef.computedWidth,
    columnMenuIconButton: columnMenuIconButton,
    columnTitleIconButtons: columnTitleIconButtons,
    headerClassName: (0, _clsx.default)(headerClassName, isLast && _gridClasses.gridClasses['columnHeader--last']),
    label: label,
    resizable: !rootProps.disableColumnResize && !!colDef.resizable,
    "data-field": colDef.field,
    columnMenu: columnMenu,
    draggableContainerProps: draggableEventHandlers,
    columnHeaderSeparatorProps: columnHeaderSeparatorProps,
    style: style
  }, mouseEventsHandlers));
}
process.env.NODE_ENV !== "production" ? GridColumnHeaderItem.propTypes = {
  // ----------------------------- Warning --------------------------------
  // | These PropTypes are generated from the TypeScript type definitions |
  // | To update them edit the TypeScript types and run "pnpm proptypes"  |
  // ----------------------------------------------------------------------
  colDef: _propTypes.default.object.isRequired,
  colIndex: _propTypes.default.number.isRequired,
  columnMenuOpen: _propTypes.default.bool.isRequired,
  disableReorder: _propTypes.default.bool,
  filterItemsCounter: _propTypes.default.number,
  hasFocus: _propTypes.default.bool,
  headerHeight: _propTypes.default.number.isRequired,
  isDragging: _propTypes.default.bool.isRequired,
  isLast: _propTypes.default.bool.isRequired,
  isResizing: _propTypes.default.bool.isRequired,
  isSiblingFocused: _propTypes.default.bool.isRequired,
  pinnedOffset: _propTypes.default.number,
  pinnedPosition: _propTypes.default.oneOf([0, 1, 2, 3]),
  separatorSide: _propTypes.default.oneOf(['left', 'right']),
  showLeftBorder: _propTypes.default.bool.isRequired,
  showRightBorder: _propTypes.default.bool.isRequired,
  sortDirection: _propTypes.default.oneOf(['asc', 'desc']),
  sortIndex: _propTypes.default.number,
  style: _propTypes.default.object,
  tabIndex: _propTypes.default.oneOf([-1, 0]).isRequired
} : void 0;
const Memoized = exports.GridColumnHeaderItem = (0, _fastMemo.fastMemo)(GridColumnHeaderItem);