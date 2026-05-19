import clsx from 'clsx';
import { gridClasses } from "../constants/index.js";
import { jsx as _jsx } from "react/jsx-runtime";
const classes = {
  root: gridClasses.scrollbarFiller,
  pinnedRight: gridClasses['scrollbarFiller--pinnedRight']
};
function GridScrollbarFillerCell({
  pinnedRight
}) {
  return /*#__PURE__*/_jsx("div", {
    role: "presentation",
    className: clsx(classes.root, pinnedRight && classes.pinnedRight)
  });
}
export { GridScrollbarFillerCell };