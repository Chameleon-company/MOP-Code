import { Component, type JSX, type ReactNode } from 'react';
type ScriptLoadedState = {
    scriptLoaded: boolean;
};
type ScriptLoadedProps = {
    children: ReactNode | ReactNode[] | Function;
};
declare class ScriptLoaded extends Component<ScriptLoadedProps, ScriptLoadedState> {
    interval: number | undefined;
    constructor(props: ScriptLoadedProps);
    setScriptLoadedCallback: () => void;
    checkIfScriptLoaded: () => void;
    componentWillUnmount(): void;
    render(): JSX.Element;
}
export default ScriptLoaded;
