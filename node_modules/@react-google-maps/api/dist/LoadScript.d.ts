import { type JSX, PureComponent, type ReactNode } from 'react';
import { type LoadScriptUrlOptions } from './utils/make-load-script-url.js';
type LoadScriptState = {
    loaded: boolean;
};
export type LoadScriptProps = LoadScriptUrlOptions & {
    children?: ReactNode | undefined;
    id: string;
    nonce?: string | undefined;
    loadingElement?: ReactNode;
    onLoad?: () => void;
    onError?: (error: Error) => void;
    onUnmount?: () => void;
    preventGoogleFontsLoading?: boolean;
};
export declare function DefaultLoadingElement(): JSX.Element;
export declare const defaultLoadScriptProps: {
    id: string;
    version: string;
};
declare class LoadScript extends PureComponent<LoadScriptProps, LoadScriptState> {
    static defaultProps: {
        id: string;
        version: string;
    };
    check: HTMLDivElement | null;
    state: {
        loaded: boolean;
    };
    cleanupCallback: () => void;
    componentDidMount(): void;
    componentDidUpdate(prevProps: LoadScriptProps): void;
    componentWillUnmount(): void;
    isCleaningUp: () => Promise<void>;
    cleanup: () => void;
    injectScript: () => void;
    getRef: (el: HTMLDivElement | null) => void;
    render(): ReactNode;
}
export default LoadScript;
