import * as React from 'react';
import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommunity } from "../models/api/gridApiCommunity.js";
import type { GridConfiguration } from "../models/configuration/gridConfiguration.js";
type GridContextProviderProps = {
  privateApiRef: RefObject<GridPrivateApiCommunity>;
  configuration: GridConfiguration;
  props: {};
  children: React.ReactNode;
};
export declare function GridContextProvider({
  privateApiRef,
  configuration,
  props,
  children
}: GridContextProviderProps): import("react/jsx-runtime").JSX.Element;
export {};