import * as React from 'react';
import type { RefObject } from '@mui/x-internals/types';
import type { GridPrivateApiCommon } from "../../models/api/gridApiCommon.js";
import type { GridPrivateApiCommunity } from "../../models/api/gridApiCommunity.js";
export declare const GridPrivateApiContext: React.Context<unknown>;
export declare function useGridPrivateApiContext<PrivateApi extends GridPrivateApiCommon = GridPrivateApiCommunity>(): RefObject<PrivateApi>;