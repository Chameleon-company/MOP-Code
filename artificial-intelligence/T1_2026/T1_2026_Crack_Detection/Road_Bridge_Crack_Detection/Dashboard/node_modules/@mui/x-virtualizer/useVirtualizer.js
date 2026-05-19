"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useVirtualizer = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var _useLazyRef = _interopRequireDefault(require("@mui/utils/useLazyRef"));
var _store = require("@mui/x-internals/store");
var _colspan = require("./features/colspan");
var _dimensions = require("./features/dimensions");
var _keyboard = require("./features/keyboard");
var _rowspan = require("./features/rowspan");
var _virtualization = require("./features/virtualization");
var _constants = require("./constants");
/* eslint-disable jsdoc/require-param-type */
/* eslint-disable jsdoc/require-param-description */
/* eslint-disable jsdoc/require-returns-type */

const FEATURES = [_dimensions.Dimensions, _virtualization.Virtualization, _colspan.Colspan, _rowspan.Rowspan, _keyboard.Keyboard];
const useVirtualizer = params => {
  const paramsWithDefault = mergeDefaults(params, _constants.DEFAULT_PARAMS);
  const store = (0, _useLazyRef.default)(() => {
    return new _store.Store(FEATURES.map(f => f.initialize(paramsWithDefault)).reduce((state, partial) => Object.assign(state, partial), {}));
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
exports.useVirtualizer = useVirtualizer;
function mergeDefaults(params, defaults) {
  const result = (0, _extends2.default)({}, params);
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