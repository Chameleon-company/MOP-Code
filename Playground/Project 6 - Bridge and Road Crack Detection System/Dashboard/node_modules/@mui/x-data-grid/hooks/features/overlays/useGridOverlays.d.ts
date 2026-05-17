import type * as React from 'react';
import type { GridApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { DataGridProcessedProps } from "../../../models/props/DataGridProps.js";
import type { GridOverlayType, GridLoadingOverlayVariant } from "./gridOverlaysInterfaces.js";
/**
 * Uses the grid state to determine which overlay to display.
 * Returns the active overlay type and the active loading overlay variant.
 */
export declare const useGridOverlays: (apiRef: React.RefObject<GridApiCommunity>, props: Pick<DataGridProcessedProps, "slotProps">) => {
  overlayType: NonNullable<GridOverlayType>;
  loadingOverlayVariant: GridLoadingOverlayVariant | null;
};