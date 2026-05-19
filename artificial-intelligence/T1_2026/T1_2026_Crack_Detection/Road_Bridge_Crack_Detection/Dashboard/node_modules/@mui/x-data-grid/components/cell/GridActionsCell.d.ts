import * as React from 'react';
import type { GridRowParams } from "../../models/params/gridRowParams.js";
import type { GridRenderCellParams } from "../../models/params/gridCellParams.js";
import { type GridMenuProps } from "../menu/GridMenu.js";
import type { GridValidRowModel, GridTreeNodeWithRender } from "../../models/gridRows.js";
interface GridActionsCellProps<R extends GridValidRowModel = any, V = any, F = V, N extends GridTreeNodeWithRender = GridTreeNodeWithRender> extends Omit<GridRenderCellParams<R, V, F, N>, 'api'> {
  api?: GridRenderCellParams['api'];
  position?: GridMenuProps['position'];
  children: React.ReactNode;
  /**
   * If true, the children passed to the component will not be validated.
   * If false, only `GridActionsCellItem` and `React.Fragment` are allowed as children.
   * Only use this prop if you know what you are doing.
   * @default false
   */
  suppressChildrenValidation?: boolean;
  /**
   * Callback to fire before the menu gets opened.
   * Use this callback to prevent the menu from opening.
   *
   * @param {GridRowParams<R>} params Row parameters.
   * @param {React.MouseEvent<HTMLElement>} event The event triggering this callback.
   * @returns {boolean} if the menu should be opened.
   */
  onMenuOpen?: (params: GridRowParams<R>, event: React.MouseEvent<HTMLElement>) => boolean;
  /**
   * Callback to fire before the menu gets closed.
   * Use this callback to prevent the menu from closing.
   *
   * @param {GridRowParams<R>} params Row parameters.
   * @param {React.MouseEvent<HTMLElement> | React.KeyboardEvent | MouseEvent | TouchEvent | undefined} event The event triggering this callback.
   * @returns {boolean} if the menu should be closed.
   */
  onMenuClose?: (params: GridRowParams<R>, event: React.MouseEvent<HTMLElement> | React.KeyboardEvent | MouseEvent | TouchEvent | undefined) => boolean;
}
declare function GridActionsCell<R extends GridValidRowModel = any, V = any, F = V, N extends GridTreeNodeWithRender = GridTreeNodeWithRender>(props: GridActionsCellProps<R, V, F, N>): import("react/jsx-runtime").JSX.Element;
declare namespace GridActionsCell {
  var propTypes: any;
}
export { GridActionsCell };
export declare const renderActionsCell: (params: GridRenderCellParams) => import("react/jsx-runtime").JSX.Element;