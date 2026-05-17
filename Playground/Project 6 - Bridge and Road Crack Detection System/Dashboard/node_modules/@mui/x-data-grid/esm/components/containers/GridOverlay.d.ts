import * as React from 'react';
import { type Theme, type SxProps } from '@mui/system';
export type GridOverlayProps = React.HTMLAttributes<HTMLDivElement> & {
  sx?: SxProps<Theme>;
};
declare const GridOverlay: React.ForwardRefExoticComponent<GridOverlayProps> | React.ForwardRefExoticComponent<React.HTMLAttributes<HTMLDivElement> & {
  sx?: SxProps<Theme>;
} & React.RefAttributes<HTMLDivElement>>;
export { GridOverlay };