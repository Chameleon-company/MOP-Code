import type { DataGridProcessedProps } from "../../models/props/DataGridProps.js";
export type OwnerState = DataGridProcessedProps;
export declare const GridRootStyles: import("@emotion/styled").StyledComponent<import("@mui/system").MUIStyledCommonProps<import("@mui/material/styles").Theme> & {
  ownerState: OwnerState;
}, Pick<import("react").DetailedHTMLProps<import("react").HTMLAttributes<HTMLDivElement>, HTMLDivElement>, keyof import("react").ClassAttributes<HTMLDivElement> | keyof import("react").HTMLAttributes<HTMLDivElement>>, {}>;
export declare const supportsColorMix: boolean;
export declare const colorMixIfSupported: (colorMixValue: string, fallback: string) => string;