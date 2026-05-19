import { useGridSelector } from "../../utils/index.js";
import { gridExpandedRowCountSelector } from "../filter/index.js";
import { gridRowCountSelector, gridRowsLoadingSelector } from "../rows/index.js";
import { gridPinnedRowsCountSelector } from "../rows/gridRowsSelector.js";
import { gridVisibleColumnDefinitionsSelector } from "../columns/index.js";
import { gridPivotActiveSelector } from "../pivoting/index.js";
/**
 * Uses the grid state to determine which overlay to display.
 * Returns the active overlay type and the active loading overlay variant.
 */
export const useGridOverlays = (apiRef, props) => {
  const totalRowCount = useGridSelector(apiRef, gridRowCountSelector);
  const visibleRowCount = useGridSelector(apiRef, gridExpandedRowCountSelector);
  const pinnedRowsCount = useGridSelector(apiRef, gridPinnedRowsCountSelector);
  const visibleColumns = useGridSelector(apiRef, gridVisibleColumnDefinitionsSelector);
  const noRows = totalRowCount === 0 && pinnedRowsCount === 0;
  const loading = useGridSelector(apiRef, gridRowsLoadingSelector);
  const pivotActive = useGridSelector(apiRef, gridPivotActiveSelector);
  const showNoRowsOverlay = !loading && noRows;
  const showNoResultsOverlay = !loading && totalRowCount > 0 && visibleRowCount === 0;
  const showNoColumnsOverlay = !loading && visibleColumns.length === 0;
  const showEmptyPivotOverlay = showNoRowsOverlay && pivotActive;
  let overlayType = null;
  let loadingOverlayVariant = null;
  if (showNoRowsOverlay) {
    overlayType = 'noRowsOverlay';
  }
  if (showNoColumnsOverlay) {
    overlayType = 'noColumnsOverlay';
  }
  if (showEmptyPivotOverlay) {
    overlayType = 'emptyPivotOverlay';
  }
  if (showNoResultsOverlay) {
    overlayType = 'noResultsOverlay';
  }
  if (loading) {
    overlayType = 'loadingOverlay';
    loadingOverlayVariant = props.slotProps?.loadingOverlay?.[noRows ? 'noRowsVariant' : 'variant'] ?? (noRows ? 'skeleton' : 'linear-progress');
  }
  return {
    overlayType: overlayType,
    loadingOverlayVariant
  };
};