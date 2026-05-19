"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useGridRowsMeta = exports.rowsMetaStateInitializer = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var _useGridApiMethod = require("../../utils/useGridApiMethod");
var _gridPaginationSelector = require("../pagination/gridPaginationSelector");
var _pipeProcessing = require("../../core/pipeProcessing");
var _gridRowsSelector = require("./gridRowsSelector");
var _gridDimensionsSelectors = require("../dimensions/gridDimensionsSelectors");
const rowsMetaStateInitializer = (state, props, apiRef) => {
  // FIXME: This should be handled in the virtualizer eventually, but there are interdependencies
  // between state initializers that need to be untangled carefully.

  const baseRowHeight = (0, _gridDimensionsSelectors.gridRowHeightSelector)(apiRef);
  const dataRowCount = (0, _gridRowsSelector.gridRowCountSelector)(apiRef);
  const pagination = (0, _gridPaginationSelector.gridPaginationSelector)(apiRef);
  const rowCount = Math.min(pagination.enabled ? pagination.paginationModel.pageSize : dataRowCount, dataRowCount);
  return (0, _extends2.default)({}, state, {
    rowsMeta: {
      currentPageTotalHeight: rowCount * baseRowHeight,
      positions: Array.from({
        length: rowCount
      }, (_, i) => i * baseRowHeight),
      pinnedTopRowsTotalHeight: 0,
      pinnedBottomRowsTotalHeight: 0
    }
  });
};

/**
 * @requires useGridPageSize (method)
 * @requires useGridPage (method)
 */
exports.rowsMetaStateInitializer = rowsMetaStateInitializer;
const useGridRowsMeta = (apiRef, _props) => {
  const virtualizer = apiRef.current.virtualizer;
  const {
    getRowHeight,
    setLastMeasuredRowIndex,
    storeRowHeightMeasurement,
    resetRowHeights,
    hydrateRowsMeta,
    observeRowHeight,
    rowHasAutoHeight,
    getRowHeightEntry,
    getLastMeasuredRowIndex
  } = virtualizer.api.rowsMeta;
  (0, _pipeProcessing.useGridRegisterPipeApplier)(apiRef, 'rowHeight', hydrateRowsMeta);
  const rowsMetaApi = {
    unstable_getRowHeight: getRowHeight,
    unstable_setLastMeasuredRowIndex: setLastMeasuredRowIndex,
    unstable_storeRowHeightMeasurement: storeRowHeightMeasurement,
    resetRowHeights
  };
  const rowsMetaPrivateApi = {
    hydrateRowsMeta,
    observeRowHeight,
    rowHasAutoHeight,
    getRowHeightEntry,
    getLastMeasuredRowIndex
  };
  (0, _useGridApiMethod.useGridApiMethod)(apiRef, rowsMetaApi, 'public');
  (0, _useGridApiMethod.useGridApiMethod)(apiRef, rowsMetaPrivateApi, 'private');
};
exports.useGridRowsMeta = useGridRowsMeta;