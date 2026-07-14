import * as React from 'react';
import { type GridOverlayProps } from "./containers/GridOverlay.js";
import type { GridLoadingOverlayVariant } from "../hooks/features/overlays/gridOverlaysInterfaces.js";
export interface GridLoadingOverlayProps extends GridOverlayProps {
  /**
   * The variant of the overlay.
   * @default 'linear-progress'
   */
  variant?: GridLoadingOverlayVariant;
  /**
   * The variant of the overlay when no rows are displayed.
   * @default 'skeleton'
   */
  noRowsVariant?: GridLoadingOverlayVariant;
}
declare const GridLoadingOverlay: React.ForwardRefExoticComponent<GridLoadingOverlayProps> | React.ForwardRefExoticComponent<GridLoadingOverlayProps & React.RefAttributes<HTMLDivElement>>;
export { GridLoadingOverlay };