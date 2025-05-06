import { PureComponent } from 'react';
import MapContext from '../../map-context.js';
export type StreetViewServiceProps = {
    /** This callback is called when the streetViewService instance has loaded. It is called with the streetViewService instance. */
    onLoad?: ((streetViewService: google.maps.StreetViewService | null) => void) | undefined;
    /** This callback is called when the component unmounts. It is called with the streetViewService instance. */
    onUnmount?: ((streetViewService: google.maps.StreetViewService | null) => void) | undefined;
};
type StreetViewServiceState = {
    streetViewService: google.maps.StreetViewService | null;
};
export declare class StreetViewService extends PureComponent<StreetViewServiceProps, StreetViewServiceState> {
    static contextType: import("react").Context<google.maps.Map | null>;
    context: React.ContextType<typeof MapContext>;
    state: {
        streetViewService: null;
    };
    setStreetViewServiceCallback: () => void;
    componentDidMount(): void;
    componentWillUnmount(): void;
    render(): null;
}
export default StreetViewService;
