import type * as React from 'react';
import type { GridRowAriaAttributesInternalHook, GridRowsOverridableMethodsInternalHook } from "./gridRowConfiguration.js";
import type { GridAggregationInternalHooks } from "./gridAggregationConfiguration.js";
import type { GridCellEditableInternalHook } from "./gridCellEditableConfiguration.js";
import type { GridCSSVariablesInterface } from "../../constants/cssVariables.js";
import type { GridPrivateApiCommon } from "../api/gridApiCommon.js";
import type { GridPrivateApiCommunity } from "../api/gridApiCommunity.js";
import type { DataGridProcessedProps } from "../props/DataGridProps.js";
import type { GridParamsOverridableMethodsInternalHook } from "./gridParamsConfiguration.js";
export interface GridAriaAttributesInternalHook {
  useGridAriaAttributes: () => React.HTMLAttributes<HTMLElement>;
}
export interface GridInternalHook<Api, Props> extends GridAriaAttributesInternalHook, GridRowAriaAttributesInternalHook, GridCellEditableInternalHook<Api, Props>, GridAggregationInternalHooks<Api, Props>, GridRowsOverridableMethodsInternalHook<Api, Props>, GridParamsOverridableMethodsInternalHook<Api> {
  useCSSVariables: () => {
    id: string;
    variables: GridCSSVariablesInterface;
  };
}
export interface GridConfiguration<Api extends GridPrivateApiCommon = GridPrivateApiCommunity, Props = DataGridProcessedProps> {
  hooks: GridInternalHook<Api, Props>;
}