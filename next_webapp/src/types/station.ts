export type StationStatus = 'Available' | 'In Use' | 'Offline';

export interface Station {
  name: string;
  lat: number;
  lng: number;
  status: StationStatus;
  image: string;
  address?: string  // can be used for the future work.
}