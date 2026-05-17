"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.Keyboard = void 0;
var _dimensions = require("./dimensions");
var _virtualization = require("./virtualization");
/* eslint-disable import/export, @typescript-eslint/no-redeclare */

const selectors = {};
const Keyboard = exports.Keyboard = {
  initialize: initializeState,
  use: useKeyboard,
  selectors
};
function initializeState(_params) {
  return {};
}
function useKeyboard(store, params, _api) {
  const getViewportPageSize = () => {
    const dimensions = _dimensions.Dimensions.selectors.dimensions(store.state);
    if (!dimensions.isReady) {
      return 0;
    }

    // TODO: Use a combination of scrollTop, dimensions.viewportInnerSize.height and rowsMeta.possitions
    // to find out the maximum number of rows that can fit in the visible part of the grid
    if (params.getRowHeight) {
      const renderContext = _virtualization.Virtualization.selectors.renderContext(store.state);
      const viewportPageSize = renderContext.lastRowIndex - renderContext.firstRowIndex;
      return Math.min(viewportPageSize - 1, params.rows.length);
    }
    const maximumPageSizeWithoutScrollBar = Math.floor(dimensions.viewportInnerSize.height / dimensions.rowHeight);
    return Math.min(maximumPageSizeWithoutScrollBar, params.rows.length);
  };
  return {
    getViewportPageSize
  };
}