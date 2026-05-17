import * as React from 'react';
import type { GridSlotProps } from "../../models/gridSlotsComponentsProps.js";
interface GridToolbarInternalProps {
  mainControls?: React.ReactNode;
  additionalExportMenuItems?: (onMenuItemClick: () => void) => React.ReactNode;
}
export type GridToolbarProps = GridSlotProps['toolbar'] & GridToolbarInternalProps;
declare function GridToolbarDivider(props: GridSlotProps['baseDivider']): import("react/jsx-runtime").JSX.Element;
declare namespace GridToolbarDivider {
  var propTypes: any;
}
declare function GridToolbarLabel(props: React.HTMLAttributes<HTMLSpanElement>): import("react/jsx-runtime").JSX.Element;
declare function GridToolbar(props: GridToolbarProps): import("react/jsx-runtime").JSX.Element;
declare namespace GridToolbar {
  var propTypes: any;
}
export { GridToolbar, GridToolbarDivider, GridToolbarLabel };