import _extends from "@babel/runtime/helpers/esm/extends";
import { GRID_STRING_COL_DEF } from "./gridStringColDef.js";
import { renderLongTextCell } from "../components/cell/GridLongTextCell.js";
import { renderEditLongTextCell } from "../components/cell/GridEditLongTextCell.js";
export const GRID_LONG_TEXT_COL_DEF = _extends({}, GRID_STRING_COL_DEF, {
  type: 'longText',
  display: 'flex',
  renderCell: renderLongTextCell,
  renderEditCell: renderEditLongTextCell
});