import { GridColumnMenuColumnsItem } from "./menuItems/GridColumnMenuColumnsItem.js";
import { GridColumnMenuFilterItem } from "./menuItems/GridColumnMenuFilterItem.js";
import { GridColumnMenuSortItem } from "./menuItems/GridColumnMenuSortItem.js";
import type { GridGenericColumnMenuProps, GridColumnMenuComponent } from "./GridColumnMenuProps.js";
export declare const GRID_COLUMN_MENU_SLOTS: {
  columnMenuSortItem: typeof GridColumnMenuSortItem;
  columnMenuFilterItem: typeof GridColumnMenuFilterItem;
  columnMenuColumnsItem: typeof GridColumnMenuColumnsItem;
};
export declare const GRID_COLUMN_MENU_SLOT_PROPS: {
  columnMenuSortItem: {
    displayOrder: number;
  };
  columnMenuFilterItem: {
    displayOrder: number;
  };
  columnMenuColumnsItem: {
    displayOrder: number;
  };
};
declare const GridGenericColumnMenu: import("react").ForwardRefExoticComponent<GridGenericColumnMenuProps> | import("react").ForwardRefExoticComponent<GridGenericColumnMenuProps & import("react").RefAttributes<HTMLUListElement>>;
declare const GridColumnMenu: GridColumnMenuComponent;
export { GridColumnMenu, GridGenericColumnMenu };