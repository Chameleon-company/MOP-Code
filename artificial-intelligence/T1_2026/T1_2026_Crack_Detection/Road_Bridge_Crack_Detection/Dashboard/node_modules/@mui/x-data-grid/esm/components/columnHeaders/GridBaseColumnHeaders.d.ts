import * as React from 'react';
import { type SxProps, type Theme } from '@mui/system';
interface GridBaseColumnHeadersProps extends React.HTMLAttributes<HTMLDivElement> {
  sx?: SxProps<Theme>;
}
export declare const GridBaseColumnHeaders: React.ForwardRefExoticComponent<GridBaseColumnHeadersProps> | React.ForwardRefExoticComponent<GridBaseColumnHeadersProps & React.RefAttributes<HTMLDivElement>>;
export {};