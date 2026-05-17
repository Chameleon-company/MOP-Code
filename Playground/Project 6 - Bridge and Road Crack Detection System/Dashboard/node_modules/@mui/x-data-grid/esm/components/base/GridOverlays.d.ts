import * as React from 'react';
import type { GridOverlayType, GridLoadingOverlayVariant } from "../../hooks/features/overlays/gridOverlaysInterfaces.js";
interface GridOverlaysProps {
  overlayType: GridOverlayType;
  loadingOverlayVariant: GridLoadingOverlayVariant | null;
}
export declare function GridOverlayWrapper(props: React.PropsWithChildren<GridOverlaysProps>): import("react/jsx-runtime").JSX.Element;
export declare namespace GridOverlayWrapper {
  var propTypes: any;
}
export {};