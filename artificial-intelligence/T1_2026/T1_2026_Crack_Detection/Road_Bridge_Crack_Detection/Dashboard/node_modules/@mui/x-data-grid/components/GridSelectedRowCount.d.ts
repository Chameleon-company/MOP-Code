import * as React from 'react';
import { type SxProps, type Theme } from '@mui/system';
interface SelectedRowCountProps {
  selectedRowCount: number;
}
type GridSelectedRowCountProps = React.HTMLAttributes<HTMLDivElement> & SelectedRowCountProps & {
  sx?: SxProps<Theme>;
};
declare const GridSelectedRowCount: React.ForwardRefExoticComponent<GridSelectedRowCountProps> | React.ForwardRefExoticComponent<React.HTMLAttributes<HTMLDivElement> & SelectedRowCountProps & {
  sx?: SxProps<Theme>;
} & React.RefAttributes<HTMLDivElement>>;
export { GridSelectedRowCount };