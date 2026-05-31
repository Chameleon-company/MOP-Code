import type { TextFieldProps } from "../../../models/gridBaseSlots.js";
import type { GridFilterInputValueProps } from "../../../models/gridFilterInputComponent.js";
export type GridFilterInputSingleSelectProps = GridFilterInputValueProps<TextFieldProps> & {
  type?: 'singleSelect';
};
declare function GridFilterInputSingleSelect(props: GridFilterInputSingleSelectProps): import("react/jsx-runtime").JSX.Element | null;
declare namespace GridFilterInputSingleSelect {
  var propTypes: any;
}
export { GridFilterInputSingleSelect };