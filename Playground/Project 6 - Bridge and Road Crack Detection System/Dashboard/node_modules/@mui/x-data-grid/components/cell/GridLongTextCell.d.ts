import * as React from 'react';
import type { GridRenderCellParams } from "../../models/params/gridCellParams.js";
import type { GridSlotProps } from "../../models/gridSlotsComponent.js";
export interface GridLongTextCellProps extends GridRenderCellParams<any, string | null> {
  /**
   * A function to customize the content rendered in the popup.
   * @param {string | null} value The cell value.
   * @returns {React.ReactNode} The content to render in the popup.
   */
  renderContent?: (value: string | null) => React.ReactNode;
  /**
   * Props passed to internal components.
   */
  slotProps?: {
    /**
     * Props passed to the root element.
     */
    root?: React.HTMLAttributes<HTMLDivElement>;
    /**
     * Props passed to the content element.
     */
    content?: React.HTMLAttributes<HTMLDivElement>;
    /**
     * Props passed to the expand button element.
     */
    expandButton?: React.ButtonHTMLAttributes<HTMLButtonElement>;
    /**
     * Props passed to the collapse button element.
     */
    collapseButton?: React.ButtonHTMLAttributes<HTMLButtonElement>;
    /**
     * Props passed to the popper element.
     */
    popper?: Partial<GridSlotProps['basePopper']>;
    /**
     * Props passed to the popper content element.
     */
    popperContent?: React.HTMLAttributes<HTMLDivElement>;
  };
}
declare function GridLongTextCell(props: GridLongTextCellProps): import("react/jsx-runtime").JSX.Element;
export { GridLongTextCell };
export declare const renderLongTextCell: (params: GridLongTextCellProps) => import("react/jsx-runtime").JSX.Element;