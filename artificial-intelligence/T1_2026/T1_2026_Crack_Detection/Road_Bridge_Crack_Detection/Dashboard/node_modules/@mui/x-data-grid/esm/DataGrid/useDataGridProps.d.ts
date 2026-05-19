import type { DataGridProcessedProps, DataGridProps } from "../models/props/DataGridProps.js";
import type { GridValidRowModel } from "../models/index.js";
export declare const useDataGridProps: <R extends GridValidRowModel>(inProps: DataGridProps<R>) => DataGridProcessedProps<R>;