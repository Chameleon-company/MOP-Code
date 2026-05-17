import type { AutocompleteProps } from "../../../models/gridBaseSlots.js";
import type { GridFilterInputValueProps } from "../../../models/gridFilterInputComponent.js";
import type { ValueOptions } from "../../../models/colDef/gridColDef.js";
export type GridFilterInputMultipleSingleSelectProps = GridFilterInputValueProps<Omit<AutocompleteProps<ValueOptions, true, false, true>, 'options'>> & {
  type?: 'singleSelect';
};
declare function GridFilterInputMultipleSingleSelect(props: GridFilterInputMultipleSingleSelectProps): import("react/jsx-runtime").JSX.Element | null;
declare namespace GridFilterInputMultipleSingleSelect {
  var propTypes: any;
}
export { GridFilterInputMultipleSingleSelect };