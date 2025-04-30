import { type ReactElement, type JSX } from 'react';
import { type UseLoadScriptOptions } from './useLoadScript.js';
export type LoadScriptNextProps = UseLoadScriptOptions & {
    loadingElement?: ReactElement | undefined;
    onLoad?: (() => void) | undefined;
    onError?: ((error: Error) => void) | undefined;
    onUnmount?: (() => void) | undefined;
    children: ReactElement;
};
declare function LoadScriptNext({ loadingElement, onLoad, onError, onUnmount, children, ...hookOptions }: LoadScriptNextProps): JSX.Element;
declare const _default: import("react").MemoExoticComponent<typeof LoadScriptNext>;
export default _default;
