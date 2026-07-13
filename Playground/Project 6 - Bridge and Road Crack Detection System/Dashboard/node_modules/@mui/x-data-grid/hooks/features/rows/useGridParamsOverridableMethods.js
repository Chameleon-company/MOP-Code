"use strict";

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useGridParamsOverridableMethods = void 0;
var React = _interopRequireWildcard(require("react"));
var _gridRowsUtils = require("./gridRowsUtils");
const useGridParamsOverridableMethods = apiRef => {
  const getCellValue = React.useCallback((id, field) => {
    const colDef = apiRef.current.getColumn(field);
    const row = apiRef.current.getRow(id);
    if (!row) {
      throw new Error(`No row with id #${id} found`);
    }
    if (!colDef || !colDef.valueGetter) {
      return row[field];
    }
    return colDef.valueGetter(row[colDef.field], row, colDef, apiRef);
  }, [apiRef]);
  const getRowValue = React.useCallback((row, colDef) => (0, _gridRowsUtils.getRowValue)(row, colDef, apiRef), [apiRef]);
  const getRowFormattedValue = React.useCallback((row, colDef) => {
    const value = getRowValue(row, colDef);
    if (!colDef || !colDef.valueFormatter) {
      return value;
    }
    return colDef.valueFormatter(value, row, colDef, apiRef);
  }, [apiRef, getRowValue]);
  return {
    getCellValue,
    getRowValue,
    getRowFormattedValue
  };
};
exports.useGridParamsOverridableMethods = useGridParamsOverridableMethods;