import * as React from 'react';
import { type SxProps, type Theme } from '@mui/material/styles';
import type { GridRenderCellParams } from "../../models/params/gridCellParams.js";
interface GridFooterCellProps extends GridRenderCellParams {
  sx?: SxProps<Theme>;
}
declare function GridFooterCellRaw(props: GridFooterCellProps): import("react/jsx-runtime").JSX.Element;
declare const GridFooterCell: React.MemoExoticComponent<typeof GridFooterCellRaw>;
export { GridFooterCell };