"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useStore = useStore;
var React = _interopRequireWildcard(require("react"));
var _shim = require("use-sync-external-store/shim");
var _withSelector = require("use-sync-external-store/shim/with-selector");
var _reactMajor = _interopRequireDefault(require("../reactMajor"));
/* We need to import the shim because React 17 does not support the `useSyncExternalStore` API.
 * More info: https://github.com/mui/mui-x/issues/18303#issuecomment-2958392341 */

/* Some tests fail in R18 with the raw useSyncExternalStore. It may be possible to make it work
 * but for now we only enable it for R19+. */
const canUseRawUseSyncExternalStore = _reactMajor.default >= 19;
const useStoreImplementation = canUseRawUseSyncExternalStore ? useStoreR19 : useStoreLegacy;
function useStore(store, selector, a1, a2, a3) {
  return useStoreImplementation(store, selector, a1, a2, a3);
}
function useStoreR19(store, selector, a1, a2, a3) {
  const getSelection = React.useCallback(() => selector(store.getSnapshot(), a1, a2, a3), [store, selector, a1, a2, a3]);
  return (0, _shim.useSyncExternalStore)(store.subscribe, getSelection, getSelection);
}
function useStoreLegacy(store, selector, a1, a2, a3) {
  return (0, _withSelector.useSyncExternalStoreWithSelector)(store.subscribe, store.getSnapshot, store.getSnapshot, state => selector(state, a1, a2, a3));
}