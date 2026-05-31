import * as React from 'react';
import type { GridSlotProps } from "../../models/gridSlotsComponent.js";
import type { GridRenderEditCellParams } from "../../models/params/gridCellParams.js";
export interface GridEditDateCellProps extends GridRenderEditCellParams {
  /**
   * Callback called when the value is changed by the user.
   * @param {React.ChangeEvent<HTMLInputElement>} event The event source of the callback.
   * @param {Date | null} newValue The value that is going to be passed to `apiRef.current.setEditCellValue`.
   * @returns {Promise<void> | void} A promise to be awaited before calling `apiRef.current.setEditCellValue`
   */
  onValueChange?: (event: React.ChangeEvent<HTMLInputElement>, newValue: Date | null) => Promise<void> | void;
  slotProps?: {
    root?: Partial<GridSlotProps['baseInput']>;
  };
}
declare function GridEditDateCell(props: GridEditDateCellProps): import("react/jsx-runtime").JSX.Element;
declare namespace GridEditDateCell {
  var propTypes: any;
}
export { GridEditDateCell };
export declare const renderEditDateCell: (params: GridRenderEditCellParams) => import("react/jsx-runtime").JSX.Element;