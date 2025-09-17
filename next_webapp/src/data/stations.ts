import type { Station } from '../types/station';

export const stations: Station[] = [
  {
    name: 'Federation Square EV Station',
    lat: -37.8136,
    lng: 144.9631,
    status: 'Available',
    image: '/img/federationSquare.avif',
  },
  {
    name: 'Melbourne Central EV Station',
    lat: -37.815,
    lng: 144.965,
    status: 'In Use',
    image: '/img/melbourneCentral.avif',
  },
  {
    name: 'Southern Cross EV Station',
    lat: -37.8183,
    lng: 144.9525,
    status: 'Available',
    image: '/img/southernCross.jpg',
  },
  {
    name: 'Docklands EV Station',
    lat: -37.8163,
    lng: 144.9441,
    status: 'Offline',
    image: '/img/docklands.jpg',
  },
  {
    name: 'Queen Victoria Market EV Station',
    lat: -37.8079,
    lng: 144.9568,
    status: 'In Use',
    image: '/img/queenVictoria.avif',
  },
  {
    name: 'Royal Botanic Gardens EV Station',
    lat: -37.8304,
    lng: 144.9796,
    status: 'Available',
    image: '/img/royalBotanic.jpg',
  },
];
