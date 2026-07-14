/* eslint-disable @typescript-eslint/no-redeclare */

export const Size = {
  EMPTY: {
    width: 0,
    height: 0
  },
  equals: (a, b) => a.width === b.width && a.height === b.height
};

// TODO

export const PinnedRows = {
  EMPTY: {
    top: [],
    bottom: []
  }
};
export const PinnedColumns = {
  EMPTY: {
    left: [],
    right: []
  }
};
export const ScrollPosition = {
  EMPTY: {
    top: 0,
    left: 0
  },
  equals: (a, b) => a.top === b.top && a.left === b.left
};
export let ScrollDirection = /*#__PURE__*/function (ScrollDirection) {
  ScrollDirection[ScrollDirection["NONE"] = 0] = "NONE";
  ScrollDirection[ScrollDirection["UP"] = 1] = "UP";
  ScrollDirection[ScrollDirection["DOWN"] = 2] = "DOWN";
  ScrollDirection[ScrollDirection["LEFT"] = 3] = "LEFT";
  ScrollDirection[ScrollDirection["RIGHT"] = 4] = "RIGHT";
  return ScrollDirection;
}({});
(function (_ScrollDirection) {
  function forDelta(dx, dy) {
    if (dx === 0 && dy === 0) {
      return ScrollDirection.NONE;
    }
    /* eslint-disable */
    if (Math.abs(dy) >= Math.abs(dx)) {
      if (dy > 0) {
        return ScrollDirection.DOWN;
      } else {
        return ScrollDirection.UP;
      }
    } else {
      if (dx > 0) {
        return ScrollDirection.RIGHT;
      } else {
        return ScrollDirection.LEFT;
      }
    }
    /* eslint-enable */
  }
  _ScrollDirection.forDelta = forDelta;
})(ScrollDirection || (ScrollDirection = {}));