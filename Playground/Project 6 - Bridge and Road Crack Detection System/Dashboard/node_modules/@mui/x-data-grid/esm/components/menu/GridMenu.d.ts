import * as React from 'react';
import type { PopperProps } from "../../models/gridBaseSlots.js";
type MenuPosition = 'bottom-end' | 'bottom-start' | 'bottom' | 'left-end' | 'left-start' | 'left' | 'right-end' | 'right-start' | 'right' | 'top-end' | 'top-start' | 'top' | undefined;
export interface GridMenuProps extends Pick<PopperProps, 'className' | 'onExited'> {
  open: boolean;
  target: HTMLElement | null;
  onClose: (event?: React.KeyboardEvent | MouseEvent | TouchEvent) => void;
  position?: MenuPosition;
  children: React.ReactNode;
}
declare function GridMenu(props: GridMenuProps): import("react/jsx-runtime").JSX.Element;
declare namespace GridMenu {
  var propTypes: any;
}
export { GridMenu };