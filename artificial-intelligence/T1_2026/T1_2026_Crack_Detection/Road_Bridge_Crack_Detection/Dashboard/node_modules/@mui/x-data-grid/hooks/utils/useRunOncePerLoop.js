"use strict";
'use client';

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useRunOncePerLoop = useRunOncePerLoop;
var React = _interopRequireWildcard(require("react"));
function useRunOncePerLoop(callback) {
  const scheduledCallbackRef = React.useRef(null);
  const schedule = React.useCallback((...args) => {
    // for robustness, a fallback in case we don't react to state updates and layoutEffect is not run
    // if we react to state updates, layoutEffect will run before microtasks
    if (!scheduledCallbackRef.current) {
      queueMicrotask(() => {
        if (scheduledCallbackRef.current) {
          scheduledCallbackRef.current();
        }
      });
    }
    scheduledCallbackRef.current = () => {
      scheduledCallbackRef.current = null;
      callback(...args);
    };
  }, [callback]);
  React.useLayoutEffect(() => {
    if (scheduledCallbackRef.current) {
      scheduledCallbackRef.current();
    }
  });
  return {
    schedule,
    cancel: () => {
      if (scheduledCallbackRef.current) {
        scheduledCallbackRef.current = null;
        return true;
      }
      return false;
    }
  };
}