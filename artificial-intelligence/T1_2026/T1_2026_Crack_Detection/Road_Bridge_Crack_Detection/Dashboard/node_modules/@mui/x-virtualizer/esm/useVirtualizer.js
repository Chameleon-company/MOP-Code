import _extends from "@babel/runtime/helpers/esm/extends";
import useLazyRef from '@mui/utils/useLazyRef';
import { Store } from '@mui/x-internals/store';
import { Colspan } from "./features/colspan.js";
import { Dimensions } from "./features/dimensions.js";
import { Keyboard } from "./features/keyboard.js";
import { Rowspan } from "./features/rowspan.js";
import { Virtualization } from "./features/virtualization/index.js";
import { DEFAULT_PARAMS } from "./constants.js";

/* eslint-disable jsdoc/require-param-type */
/* eslint-disable jsdoc/require-param-description */
/* eslint-disable jsdoc/require-returns-type */

const FEATURES = [Dimensions, Virtualization, Colspan, Rowspan, Keyboard];
export const useVirtualizer = params => {
  const paramsWithDefault = mergeDefaults(params, DEFAULT_PARAMS);
  const store = useLazyRef(() => {
    return new Store(FEATURES.map(f => f.initialize(paramsWithDefault)).reduce((state, partial) => Object.assign(state, partial), {}));
  }).current;
  const api = {};
  for (const feature of FEATURES) {
    Object.assign(api, feature.use(store, paramsWithDefault, api));
  }
  return {
    store,
    api
  };
};
function mergeDefaults(params, defaults) {
  const result = _extends({}, params);
  for (const key in defaults) {
    if (Object.hasOwn(defaults, key)) {
      const value = defaults[key];
      if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
        result[key] = mergeDefaults(params[key] ?? {}, value);
      } else {
        result[key] = params[key] ?? value;
      }
    }
  }
  return result;
}