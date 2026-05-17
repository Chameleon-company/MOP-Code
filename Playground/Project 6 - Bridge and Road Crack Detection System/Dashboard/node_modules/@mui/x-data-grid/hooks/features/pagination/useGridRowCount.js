"use strict";
'use client';

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useGridRowCount = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var React = _interopRequireWildcard(require("react"));
var _useLazyRef = _interopRequireDefault(require("@mui/utils/useLazyRef"));
var _store = require("@mui/x-internals/store");
var _filter = require("../filter");
var _utils = require("../../utils");
var _pipeProcessing = require("../../core/pipeProcessing");
var _gridPaginationSelector = require("./gridPaginationSelector");
const useGridRowCount = (apiRef, props) => {
  const logger = (0, _utils.useGridLogger)(apiRef, 'useGridRowCount');
  const previousPageSize = (0, _useLazyRef.default)(() => (0, _gridPaginationSelector.gridPaginationModelSelector)(apiRef).pageSize);
  apiRef.current.registerControlState({
    stateId: 'paginationRowCount',
    propModel: props.rowCount,
    propOnChange: props.onRowCountChange,
    stateSelector: _gridPaginationSelector.gridPaginationRowCountSelector,
    changeEvent: 'rowCountChange'
  });

  /**
   * API METHODS
   */
  const setRowCount = React.useCallback(newRowCount => {
    const rowCountState = (0, _gridPaginationSelector.gridPaginationRowCountSelector)(apiRef);
    if (rowCountState === newRowCount) {
      return;
    }
    logger.debug("Setting 'rowCount' to", newRowCount);
    apiRef.current.setState(state => (0, _extends2.default)({}, state, {
      pagination: (0, _extends2.default)({}, state.pagination, {
        rowCount: newRowCount
      })
    }));
  }, [apiRef, logger]);
  const paginationRowCountApi = {
    setRowCount
  };
  (0, _utils.useGridApiMethod)(apiRef, paginationRowCountApi, 'public');

  /**
   * PRE-PROCESSING
   */
  const stateExportPreProcessing = React.useCallback((prevState, context) => {
    const exportedRowCount = (0, _gridPaginationSelector.gridPaginationRowCountSelector)(apiRef);
    const shouldExportRowCount =
    // Always export if the `exportOnlyDirtyModels` property is not activated
    !context.exportOnlyDirtyModels ||
    // Always export if the `rowCount` is controlled
    props.rowCount != null ||
    // Always export if the `rowCount` has been initialized
    props.initialState?.pagination?.rowCount != null;
    if (!shouldExportRowCount) {
      return prevState;
    }
    return (0, _extends2.default)({}, prevState, {
      pagination: (0, _extends2.default)({}, prevState.pagination, {
        rowCount: exportedRowCount
      })
    });
  }, [apiRef, props.rowCount, props.initialState?.pagination?.rowCount]);
  const stateRestorePreProcessing = React.useCallback((params, context) => {
    const restoredRowCount = context.stateToRestore.pagination?.rowCount ? context.stateToRestore.pagination.rowCount : (0, _gridPaginationSelector.gridPaginationRowCountSelector)(apiRef);
    apiRef.current.setState(state => (0, _extends2.default)({}, state, {
      pagination: (0, _extends2.default)({}, state.pagination, {
        rowCount: restoredRowCount
      })
    }));
    return params;
  }, [apiRef]);
  (0, _pipeProcessing.useGridRegisterPipeProcessor)(apiRef, 'exportState', stateExportPreProcessing);
  (0, _pipeProcessing.useGridRegisterPipeProcessor)(apiRef, 'restoreState', stateRestorePreProcessing);

  /**
   * EVENTS
   */
  const handlePaginationModelChange = React.useCallback(model => {
    if (props.paginationMode === 'client' || !previousPageSize.current) {
      return;
    }
    if (model.pageSize !== previousPageSize.current) {
      previousPageSize.current = model.pageSize;
      const rowCountState = (0, _gridPaginationSelector.gridPaginationRowCountSelector)(apiRef);
      if (rowCountState === -1) {
        // Row count unknown and page size changed, reset the page
        apiRef.current.setPage(0);
      }
    }
  }, [props.paginationMode, previousPageSize, apiRef]);
  (0, _utils.useGridEvent)(apiRef, 'paginationModelChange', handlePaginationModelChange);

  /**
   * EFFECTS
   */
  React.useEffect(() => {
    if (props.paginationMode === 'server' && props.rowCount != null) {
      apiRef.current.setRowCount(props.rowCount);
    }
  }, [apiRef, props.paginationMode, props.rowCount]);
  (0, _store.useStoreEffect)(
  // typings not supported currently, but methods work
  apiRef.current.store, () => {
    const isLastPage = (0, _gridPaginationSelector.gridPaginationMetaSelector)(apiRef).hasNextPage === false;
    if (isLastPage) {
      return true;
    }
    if (props.paginationMode === 'client') {
      return (0, _filter.gridFilteredTopLevelRowCountSelector)(apiRef);
    }
    return undefined;
  }, (_, isLastPageOrRowCount) => {
    if (isLastPageOrRowCount === true && (0, _gridPaginationSelector.gridPaginationRowCountSelector)(apiRef) !== -1) {
      const visibleTopLevelRowCount = (0, _filter.gridFilteredTopLevelRowCountSelector)(apiRef);
      const paginationModel = (0, _gridPaginationSelector.gridPaginationModelSelector)(apiRef);
      apiRef.current.setRowCount(paginationModel.pageSize * paginationModel.page + visibleTopLevelRowCount);
    } else if (typeof isLastPageOrRowCount === 'number') {
      apiRef.current.setRowCount(isLastPageOrRowCount);
    }
  });
  React.useEffect(() => {
    if (props.paginationMode === 'client') {
      apiRef.current.setRowCount((0, _filter.gridFilteredTopLevelRowCountSelector)(apiRef));
    }
  }, [apiRef, props.paginationMode]);
};
exports.useGridRowCount = useGridRowCount;