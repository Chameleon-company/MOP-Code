import type * as React from 'react';
import type { GridColDef } from "../../../models/colDef/gridColDef.js";
export interface GridColumnMenuItemProps {
  colDef: GridColDef;
  onClick: (event: React.MouseEvent<any>) => void;
  [key: string]: any;
}