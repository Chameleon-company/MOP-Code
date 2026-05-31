'use client';

import _extends from "@babel/runtime/helpers/esm/extends";
import * as React from 'react';
export const propsStateInitializer = (state, props) => {
  return _extends({}, state, {
    props: {
      listView: props.listView,
      getRowId: props.getRowId,
      isCellEditable: props.isCellEditable,
      isRowSelectable: props.isRowSelectable,
      dataSource: props.dataSource
    }
  });
};
export const useGridProps = (apiRef, props) => {
  const isFirstRender = React.useRef(true);
  React.useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    apiRef.current.setState(state => _extends({}, state, {
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