import * as React from 'react';
import type { GridRenderEditCellParams } from "../../models/params/gridCellParams.js";
import type { GridSlotProps } from "../../models/gridSlotsComponent.js";
export interface GridEditLongTextCellProps extends GridRenderEditCellParams<any, string | null> {
  debounceMs?: number;
  /**
   * Callback called when the value is changed by the user.
   * @param {React.ChangeEvent<HTMLTextAreaElement>} event The event source of the callback.
   * @param {string} newValue The value that is going to be passed to `apiRef.current.setEditCellValue`.
   * @returns {Promise<void> | void} A promise to be awaited before calling `apiRef.current.setEditCellValue`
   */
  onValueChange?: (event: React.ChangeEvent<HTMLTextAreaElement>, newValue: string) => Promise<void> | void;
  /**
   * Props passed to internal components.
   */
  slotProps?: {
    /**
     * Props passed to the root element.
     */
    root?: React.HTMLAttributes<HTMLDivElement>;
    /**
     * Props passed to the value element.
     */
    value?: React.HTMLAttributes<HTMLDivElement>;
    /**
     * Props passed to the popper element.
     */
    popper?: Partial<GridSlotProps['basePopper']>;
    /**
     * Props passed to the popper content element.
     */
    popperContent?: React.HTMLAttributes<HTMLDivElement>;
    /**
     * Props passed to the textarea element.
     */
    textarea?: Partial<GridSlotProps['baseTextarea']>;
  };
}
declare function GridEditLongTextCell(props: GridEditLongTextCellProps): import("react/jsx-runtime").JSX.Element;
export { GridEditLongTextCell };
export declare const renderEditLongTextCell: (params: GridEditLongTextCellProps) => import("react/jsx-runtime").JSX.Element;