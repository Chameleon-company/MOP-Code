import type { AutocompleteProps } from "../../../models/gridBaseSlots.js";
import type { GridFilterInputValueProps } from "../../../models/gridFilterInputComponent.js";
export type GridFilterInputMultipleValueProps = GridFilterInputValueProps<Omit<AutocompleteProps<string, true, false, true>, 'options'>> & {
  type?: 'text' | 'number' | 'date' | 'datetime-local';
};
declare function GridFilterInputMultipleValue(props: GridFilterInputMultipleValueProps): import("react/jsx-runtime").JSX.Element;
declare namespace GridFilterInputMultipleValue {
  var propTypes: any;
}
export { GridFilterInputMultipleValue };