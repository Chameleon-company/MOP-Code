import { type GridFilterInputDateProps } from "../components/panel/filterPanel/GridFilterInputDate.js";
import type { GridFilterOperator } from "../models/gridFilterOperator.js";
export declare const getGridDateOperators: (showTime?: boolean) => GridFilterOperator<any, Date, any, GridFilterInputDateProps>[];