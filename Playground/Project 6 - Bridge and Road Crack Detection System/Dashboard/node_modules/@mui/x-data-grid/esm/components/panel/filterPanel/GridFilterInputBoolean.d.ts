import type { TextFieldProps } from "../../../models/gridBaseSlots.js";
import type { GridFilterInputValueProps } from "../../../models/gridFilterInputComponent.js";
export type GridFilterInputBooleanProps = GridFilterInputValueProps<TextFieldProps>;
declare function GridFilterInputBoolean(props: GridFilterInputBooleanProps): import("react/jsx-runtime").JSX.Element;
declare namespace GridFilterInputBoolean {
  var propTypes: any;
}
export declare function sanitizeFilterItemValue(value: any): boolean | undefined;
export { GridFilterInputBoolean };