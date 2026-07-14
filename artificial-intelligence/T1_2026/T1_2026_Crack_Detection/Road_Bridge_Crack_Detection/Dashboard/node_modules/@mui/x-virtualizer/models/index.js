"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
var _core = require("./core");
Object.keys(_core).forEach(function (key) {
  if (key === "default" || key === "__esModule") return;
  if (key in exports && exports[key] === _core[key]) return;
  Object.defineProperty(exports, key, {
    enumerable: true,
    get: function () {
      return _core[key];
    }
  });
});
var _colspan = require("./colspan");
Object.keys(_colspan).forEach(function (key) {
  if (key === "default" || key === "__esModule") return;
  if (key in exports && exports[key] === _colspan[key]) return;
  Object.defineProperty(exports, key, {
    enumerable: true,
    get: function () {
      return _colspan[key];
    }
  });
});
var _dimensions = require("./dimensions");
Object.keys(_dimensions).forEach(function (key) {
  if (key === "default" || key === "__esModule") return;
  if (key in exports && exports[key] === _dimensions[key]) return;
  Object.defineProperty(exports, key, {
    enumerable: true,
    get: function () {
      return _dimensions[key];
    }
  });
});
var _rowspan = require("./rowspan");
Object.keys(_rowspan).forEach(function (key) {
  if (key === "default" || key === "__esModule") return;
  if (key in exports && exports[key] === _rowspan[key]) return;
  Object.defineProperty(exports, key, {
    enumerable: true,
    get: function () {
      return _rowspan[key];
    }
  });
});