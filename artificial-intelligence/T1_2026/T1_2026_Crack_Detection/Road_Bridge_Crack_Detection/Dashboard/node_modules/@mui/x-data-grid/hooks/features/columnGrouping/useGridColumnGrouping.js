"use strict";
'use client';

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useGridColumnGrouping = exports.columnGroupsStateInitializer = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var React = _interopRequireWildcard(require("react"));
var _gridColumnGroupsSelector = require("./gridColumnGroupsSelector");
var _useGridApiMethod = require("../../utils/useGridApiMethod");
var _gridColumnGroupsUtils = require("./gridColumnGroupsUtils");
var _useGridEvent = require("../../utils/useGridEvent");
var _columns = require("../columns");
const columnGroupsStateInitializer = (state, props, apiRef) => {
  apiRef.current.caches.columnGrouping = {
    lastColumnGroupingModel: props.columnGroupingModel
  };
  if (!props.columnGroupingModel) {
    return (0, _extends2.default)({}, state, {
      columnGrouping: undefined
    });
  }
  const columnFields = (0, _columns.gridColumnFieldsSelector)(apiRef);
  const visibleColumnFields = (0, _columns.gridVisibleColumnFieldsSelector)(apiRef);
  const groupLookup = (0, _gridColumnGroupsUtils.createGroupLookup)(props.columnGroupingModel ?? []);
  const unwrappedGroupingModel = (0, _gridColumnGroupsUtils.unwrapGroupingColumnModel)(props.columnGroupingModel ?? []);
  const columnGroupsHeaderStructure = (0, _gridColumnGroupsUtils.getColumnGroupsHeaderStructure)(columnFields, unwrappedGroupingModel, apiRef.current.state.pinnedColumns ?? {});
  const maxDepth = visibleColumnFields.length === 0 ? 0 : Math.max(...visibleColumnFields.map(field => unwrappedGroupingModel[field]?.length ?? 0));
  return (0, _extends2.default)({}, state, {
    columnGrouping: {
      lookup: groupLookup,
      unwrappedGroupingModel,
      headerStructure: columnGroupsHeaderStructure,
      maxDepth
    }
  });
};

/**
 * @requires useGridColumns (method, event)
 * @requires useGridParamsApi (method)
 */
exports.columnGroupsStateInitializer = columnGroupsStateInitializer;
const useGridColumnGrouping = (apiRef, props) => {
  /**
   * API METHODS
   */
  const getColumnGroupPath = React.useCallback(field => {
    const unwrappedGroupingModel = (0, _gridColumnGroupsSelector.gridColumnGroupsUnwrappedModelSelector)(apiRef);
    return unwrappedGroupingModel[field] ?? [];
  }, [apiRef]);
  const getAllGroupDetails = React.useCallback(() => {
    const columnGroupLookup = (0, _gridColumnGroupsSelector.gridColumnGroupsLookupSelector)(apiRef);
    return columnGroupLookup;
  }, [apiRef]);
  const columnGroupingApi = {
    getColumnGroupPath,
    getAllGroupDetails
  };
  (0, _useGridApiMethod.useGridApiMethod)(apiRef, columnGroupingApi, 'public');
  const handleColumnIndexChange = React.useCallback(() => {
    const unwrappedGroupingModel = (0, _gridColumnGroupsUtils.unwrapGroupingColumnModel)(props.columnGroupingModel ?? []);
    apiRef.current.setState(state => {
      const orderedFields = state.columns?.orderedFields ?? [];
      const pinnedColumns = state.pinnedColumns ?? {};
      const columnGroupsHeaderStructure = (0, _gridColumnGroupsUtils.getColumnGroupsHeaderStructure)(orderedFields, unwrappedGroupingModel, pinnedColumns);
      return (0, _extends2.default)({}, state, {
        columnGrouping: (0, _extends2.default)({}, state.columnGrouping, {
          headerStructure: columnGroupsHeaderStructure
        })
      });
    });
  }, [apiRef, props.columnGroupingModel]);
  const updateColumnGroupingState = React.useCallback(columnGroupingModel => {
    if (!columnGroupingModel && !apiRef.current.caches.columnGrouping.lastColumnGroupingModel) {
      return;
    }
    apiRef.current.caches.columnGrouping.lastColumnGroupingModel = columnGroupingModel;
    // @ts-expect-error Move this logic to `Pro` package
    const pinnedColumns = apiRef.current.getPinnedColumns?.() ?? {};
    const columnFields = (0, _columns.gridColumnFieldsSelector)(apiRef);
    const visibleColumnFields = (0, _columns.gridVisibleColumnFieldsSelector)(apiRef);
    const groupLookup = (0, _gridColumnGroupsUtils.createGroupLookup)(columnGroupingModel ?? []);
    const unwrappedGroupingModel = (0, _gridColumnGroupsUtils.unwrapGroupingColumnModel)(columnGroupingModel ?? []);
    const columnGroupsHeaderStructure = (0, _gridColumnGroupsUtils.getColumnGroupsHeaderStructure)(columnFields, unwrappedGroupingModel, pinnedColumns);
    const maxDepth = visibleColumnFields.length === 0 ? 0 : Math.max(...visibleColumnFields.map(field => unwrappedGroupingModel[field]?.length ?? 0));
    apiRef.current.setState(state => {
      return (0, _extends2.default)({}, state, {
        columnGrouping: {
          lookup: groupLookup,
          unwrappedGroupingModel,
          headerStructure: columnGroupsHeaderStructure,
          maxDepth
        }
      });
    });
  }, [apiRef]);
  (0, _useGridEvent.useGridEvent)(apiRef, 'columnIndexChange', handleColumnIndexChange);
  (0, _useGridEvent.useGridEvent)(apiRef, 'columnsChange', () => {
    updateColumnGroupingState(props.columnGroupingModel);
  });
  (0, _useGridEvent.useGridEvent)(apiRef, 'columnVisibilityModelChange', () => {
    updateColumnGroupingState(props.columnGroupingModel);
  });

  /**
   * EFFECTS
   */
  React.useEffect(() => {
    if (props.columnGroupingModel === apiRef.current.caches.columnGrouping.lastColumnGroupingModel) {
      return;
    }
    updateColumnGroupingState(props.columnGroupingModel);
  }, [apiRef, updateColumnGroupingState, props.columnGroupingModel]);
};
exports.useGridColumnGrouping = useGridColumnGrouping;