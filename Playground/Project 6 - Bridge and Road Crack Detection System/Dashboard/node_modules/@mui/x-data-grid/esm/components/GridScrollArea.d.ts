import type { RefObject } from '@mui/x-internals/types';
import type { GridScrollParams } from "../models/params/gridScrollParams.js";
interface ScrollAreaProps {
  scrollDirection: 'left' | 'right' | 'up' | 'down';
  scrollPosition: RefObject<GridScrollParams>;
}
declare function GridScrollAreaWrapper(props: ScrollAreaProps): import("react/jsx-runtime").JSX.Element | null;
export declare const GridScrollArea: typeof GridScrollAreaWrapper;
export {};