import type { GridSlotsComponent } from "../../../models/index.js";
export type GridOverlayType = keyof Pick<GridSlotsComponent, 'noColumnsOverlay' | 'noRowsOverlay' | 'noResultsOverlay' | 'loadingOverlay'> | null;
export type GridLoadingOverlayVariant = 'circular-progress' | 'linear-progress' | 'skeleton';