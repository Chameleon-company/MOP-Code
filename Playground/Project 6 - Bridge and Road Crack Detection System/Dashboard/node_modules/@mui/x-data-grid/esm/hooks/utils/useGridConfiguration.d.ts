import type { GridConfiguration } from "../../models/configuration/gridConfiguration.js";
import type { GridPrivateApiCommon } from "../../models/api/gridApiCommon.js";
import type { GridPrivateApiCommunity } from "../../models/api/gridApiCommunity.js";
export declare const useGridConfiguration: <Api extends GridPrivateApiCommon = GridPrivateApiCommunity>() => GridConfiguration<Api>;