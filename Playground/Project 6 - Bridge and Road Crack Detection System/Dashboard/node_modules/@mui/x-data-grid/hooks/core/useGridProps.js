"use strict";
'use client';

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard").default;
var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault").default;
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.useGridProps = exports.propsStateInitializer = void 0;
var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));
var React = _interopRequireWildcard(require("react"));
const propsStateInitializer = (state, props) => {
  return (0, _extends2.default)({}, state, {
    props: {
      listView: props.listView,
      getRowId: props.getRowId,
      isCellEditable: props.isCellEditable,
      isRowSelectable: props.isRowSelectable,
      dataSource: props.dataSource
    }
  });
};
exports.propsStateInitializer = propsStateInitializer;
const useGridProps = (apiRef, props) => {
  const isFirstRender = React.useRef(true);
  React.useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    apiRef.current.setState(state => (0, _extends2.default)({}, state, {
      props: {
        listView: props.listView,
        getRowId: props.getRowId,
        isCellEditable: props.isCellEditable,
        isRowSelectable: props.isRowSelectable,
        dataSource: props.dataSource
      }
    }));
  }, [apiRef, props.listView, props.getRowId, props.isCellEditable, props.isRowSelectable, props.dataSource]);
};
exports.useGridProps = useGridProps;