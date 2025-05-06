import type { Library } from '@googlemaps/js-api-loader';
export type Libraries = Library[];
export type LoadScriptUrlOptions = {
    googleMapsApiKey: string | '';
    googleMapsClientId?: string | undefined;
    version?: string | undefined;
    language?: string | undefined;
    region?: string | undefined;
    libraries?: Libraries | undefined;
    channel?: string | undefined;
    mapIds?: string[] | undefined;
    authReferrerPolicy?: 'origin' | undefined;
};
export declare function makeLoadScriptUrl({ googleMapsApiKey, googleMapsClientId, version, language, region, libraries, channel, mapIds, authReferrerPolicy, }: LoadScriptUrlOptions): string;
