import * as React from 'react';
export interface GridColumnHeaderTitleProps {
  label: string;
  columnWidth: number;
  description?: React.ReactNode;
}
declare function GridColumnHeaderTitle(props: GridColumnHeaderTitleProps): import("react/jsx-runtime").JSX.Element;
declare namespace GridColumnHeaderTitle {
  var propTypes: any;
}
export { GridColumnHeaderTitle };