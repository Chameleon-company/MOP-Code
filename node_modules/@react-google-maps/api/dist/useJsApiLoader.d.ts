import type { LoadScriptUrlOptions } from './utils/make-load-script-url.js';
export type UseLoadScriptOptions = LoadScriptUrlOptions & {
    id?: string | undefined;
    nonce?: string | undefined;
    preventGoogleFontsLoading?: boolean | undefined;
};
export declare function useJsApiLoader({ id, version, nonce, googleMapsApiKey, language, region, libraries, preventGoogleFontsLoading, mapIds, authReferrerPolicy, }: UseLoadScriptOptions): {
    isLoaded: boolean;
    loadError: Error | undefined;
};
