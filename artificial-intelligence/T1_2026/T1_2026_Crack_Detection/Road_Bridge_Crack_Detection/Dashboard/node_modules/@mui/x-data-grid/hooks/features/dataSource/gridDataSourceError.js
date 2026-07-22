"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.GridUpdateRowError = exports.GridGetRowsError = void 0;
class GridGetRowsError extends Error {
  /**
   * The parameters used in the failed request
   */

  /**
   * The original error that caused this error
   */

  constructor(options) {
    super(options.message);
    this.name = 'GridGetRowsError';
    this.params = options.params;
    this.cause = options.cause;
  }
}
exports.GridGetRowsError = GridGetRowsError;
class GridUpdateRowError extends Error {
  /**
   * The parameters used in the failed request
   */

  /**
   * The original error that caused this error
   */

  constructor(options) {
    super(options.message);
    this.name = 'GridUpdateRowError';
    this.params = options.params;
    this.cause = options.cause;
  }
}
exports.GridUpdateRowError = GridUpdateRowError;