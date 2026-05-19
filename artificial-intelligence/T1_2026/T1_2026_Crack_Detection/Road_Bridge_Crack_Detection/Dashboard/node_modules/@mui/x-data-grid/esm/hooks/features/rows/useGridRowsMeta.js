import _extends from "@babel/runtime/helpers/esm/extends";
import { useGridApiMethod } from "../../utils/useGridApiMethod.js";
import { gridPaginationSelector } from "../pagination/gridPaginationSelector.js";
import { useGridRegisterPipeApplier } from "../../core/pipeProcessing/index.js";
import { gridRowCountSelector } from "./gridRowsSelector.js";
import { gridRowHeightSelector } from "../dimensions/gridDimensionsSelectors.js";
export const rowsMetaStateInitializer = (state, props, apiRef) => {
  // FIXME: This should be handled in the virtualizer eventually, but there are interdependencies
  // between state initializers that need to be untangled carefully.

  const baseRowHeight = gridRowHeightSelector(apiRef);
  const dataRowCount = gridRowCountSelector(apiRef);
  const pagination = gridPaginationSelector(apiRef);
  const rowCount = Math.min(pagination.enabled ? pagination.paginationModel.pageSize : dataRowCount, dataRowCount);
  return _extends({}, state, {
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
export const useGridRowsMeta = (apiRef, _props) => {
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
  useGridRegisterPipeApplier(apiRef, 'rowHeight', hydrateRowsMeta);
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
  useGridApiMethod(apiRef, rowsMetaApi, 'public');
  useGridApiMethod(apiRef, rowsMetaPrivateApi, 'private');
};