import type { GridFilterInputValueProps } from "../../../models/gridFilterInputComponent.js";
import type { TextFieldProps } from "../../../models/gridBaseSlots.js";
export type GridFilterInputDateProps = GridFilterInputValueProps<TextFieldProps> & {
  type?: 'date' | 'datetime-local';
};
declare function GridFilterInputDate(props: GridFilterInputDateProps): import("react/jsx-runtime").JSX.Element;
declare namespace GridFilterInputDate {
  var propTypes: any;
}
export { GridFilterInputDate };