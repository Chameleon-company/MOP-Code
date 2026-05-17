"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useGridColumnSpanning = void 0;
var _useGridApiMethod = require("../../utils/useGridApiMethod");
var _useGridEvent = require("../../utils/useGridEvent");
/**
 * @requires useGridColumns (method, event)
 * @requires useGridParamsApi (method)
 */
const useGridColumnSpanning = apiRef => {
  const resetColSpan = () => {
    return apiRef.current.virtualizer.api.resetColSpan();
  };
  const getCellColSpanInfo = (...params) => {
    return apiRef.current.virtualizer.api.getCellColSpanInfo(...params);
  };
  const calculateColSpan = (...params) => {
    apiRef.current.virtualizer.api.calculateColSpan(...params);
  };
  const columnSpanningPublicApi = {
    unstable_getCellColSpanInfo: getCellColSpanInfo
  };
  const columnSpanningPrivateApi = {
    resetColSpan,
    calculateColSpan
  };
  (0, _useGridApiMethod.useGridApiMethod)(apiRef, columnSpanningPublicApi, 'public');
  (0, _useGridApiMethod.useGridApiMethod)(apiRef, columnSpanningPrivateApi, 'private');
  (0, _useGridEvent.useGridEvent)(apiRef, 'columnOrderChange', resetColSpan);
};
exports.useGridColumnSpanning = useGridColumnSpanning;