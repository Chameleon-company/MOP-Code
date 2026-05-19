import * as React from 'react';
import type { DataGridProcessedProps } from "../../models/props/DataGridProps.js";
import type { GridOverlayType, GridLoadingOverlayVariant } from "../../hooks/features/overlays/gridOverlaysInterfaces.js";
type OwnerState = Pick<DataGridProcessedProps, 'classes'> & {
  hasScrollX: boolean;
  hasPinnedRight: boolean;
  overlayType: GridOverlayType;
  loadingOverlayVariant: GridLoadingOverlayVariant | null;
};
export declare const GridMainContainer: React.ForwardRefExoticComponent<React.PropsWithChildren<{
  className: string;
  ownerState: OwnerState;
}>> | React.ForwardRefExoticComponent<{
  className: string;
  ownerState: OwnerState;
} & {
  children?: React.ReactNode | undefined;
} & React.RefAttributes<HTMLDivElement>>;
export {};