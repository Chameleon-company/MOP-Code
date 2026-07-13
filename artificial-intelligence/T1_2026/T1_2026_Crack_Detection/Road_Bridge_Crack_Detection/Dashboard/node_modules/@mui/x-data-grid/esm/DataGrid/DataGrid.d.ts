import * as React from 'react';
import type { DataGridProps } from "../models/props/DataGridProps.js";
import type { GridValidRowModel } from "../models/gridRows.js";
export type { GridSlotsComponent as GridSlots } from "../models/index.js";
interface DataGridComponent {
  <R extends GridValidRowModel = any>(props: DataGridProps<R> & React.RefAttributes<HTMLDivElement>): React.JSX.Element;
  propTypes?: any;
}
/**
 * Features:
 * - [DataGrid](https://mui.com/x/react-data-grid/features/)
 *
 * API:
 * - [DataGrid API](https://mui.com/x/api/data-grid/data-grid/)
 */
export declare const DataGrid: DataGridComponent;