import { type LoadScriptUrlOptions } from './utils/make-load-script-url.js';
export type UseLoadScriptOptions = LoadScriptUrlOptions & {
    id?: string | undefined;
    nonce?: string | undefined;
    preventGoogleFontsLoading?: boolean | undefined;
};
export declare function useLoadScript({ id, version, nonce, googleMapsApiKey, googleMapsClientId, language, region, libraries, preventGoogleFontsLoading, channel, mapIds, authReferrerPolicy, }: UseLoadScriptOptions): {
    isLoaded: boolean;
    loadError: Error | undefined;
    url: string;
};
