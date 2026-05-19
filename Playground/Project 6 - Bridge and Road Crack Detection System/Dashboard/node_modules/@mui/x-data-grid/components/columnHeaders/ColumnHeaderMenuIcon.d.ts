import * as React from 'react';
import type { GridStateColDef } from "../../models/colDef/gridColDef.js";
export interface ColumnHeaderMenuIconProps {
  colDef: GridStateColDef;
  columnMenuId: string;
  columnMenuButtonId: string;
  open: boolean;
  iconButtonRef: React.RefObject<HTMLButtonElement | null>;
}
export declare const ColumnHeaderMenuIcon: React.MemoExoticComponent<(props: ColumnHeaderMenuIconProps) => import("react/jsx-runtime").JSX.Element>;