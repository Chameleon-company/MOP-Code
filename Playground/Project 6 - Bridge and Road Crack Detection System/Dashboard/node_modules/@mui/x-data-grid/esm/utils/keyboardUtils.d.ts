import type * as React from 'react';
export declare function isPrintableKey(event: React.KeyboardEvent<HTMLElement>): boolean;
export declare const isNavigationKey: (key: string) => boolean;
export declare const isKeyboardEvent: (event: any) => event is React.KeyboardEvent<HTMLElement>;
export declare const isHideMenuKey: (key: React.KeyboardEvent["key"]) => key is "Tab" | "Escape";
export declare function isPasteShortcut(event: React.KeyboardEvent): boolean;
export declare function isCopyShortcut(event: KeyboardEvent): boolean;
export declare function isUndoShortcut(event: React.KeyboardEvent): boolean;
export declare function isRedoShortcut(event: React.KeyboardEvent): boolean;