import { FacilitySectionData } from "../types/facilities";


export const communitySectionsYouOwn: FacilitySectionData[] = [
  {
    id: "recreation",
    title: "Recreational and Sporting Facilities",
    subtitle: "Watch AFL at the MCG or Marvel Stadium or relax at Albert Park.",
    tone: "green",
    items: [
      {
        name: "Melbourne Cricket Ground (MCG)",
        href: "https://www.mcg.org.au/",
        image: "/images/facilities/mcg.jpg",
        alt: "Melbourne Cricket Ground",
      },
      {
        name: "Marvel Stadium",
        href: "https://marvelstadium.com.au/",
        image: "/images/facilities/marvel-stadium.jpg",
        alt: "Marvel Stadium",
      },
      {
        name: "Rod Laver Arena",
        href: "https://www.rodlaverarena.com.au/",
        image: "/images/facilities/rod-laver-arena.jpg",
        alt: "Rod Laver Arena",
      },
      {
        name: "Albert Park",
        href: "https://www.parks.vic.gov.au/places-to-see/parks/albert-park",
        image: "/images/facilities/albert-park.jpg",
        alt: "Albert Park Lake",
      },
    ],
  },
  {
    id: "religious",
    title: "Religious Spaces",
    subtitle: "Visit Melbourne’s various religious sites.",
    tone: "blue",
    items: [
      {
        name: "St Patrick's Cathedral",
        href: "https://melbournecatholic.org/places/st-patricks-cathedral",
        image: "/images/facilities/st-patricks.jpg",
        alt: "St Patrick's Cathedral",
      },
      {
        name: "St Paul’s Cathedral",
        href: "https://cathedral.org.au/",
        image: "/images/facilities/st-pauls.jpg",
        alt: "St Paul’s Cathedral",
      },
      {
        name: "St Francis Church",
        href: "https://www.stfrancismelbourne.org.au/",
        image: "/images/facilities/st-francis.jpg",
        alt: "St Francis Church",
      },
      {
        name: "Queen Street Masjid",
        href: "#", // replace when you have the official site
        image: "/images/facilities/queen-st-masjid.jpg",
        alt: "Queen Street Mosque",
      },
    ],
  },
  {
    id: "kids",
    title: "Kid’s Spaces",
    subtitle: "Fun for families at Chipmunks Playland, PlanetKids Play Centre and more.",
    tone: "green",
    items: [
      {
        name: "Chipmunks Playland",
        href: "https://chipmunks.com.au/",
        image: "/images/facilities/chipmunks.jpg",
        alt: "Chipmunks Playland",
      },
      {
        name: "PlanetKids Play Centre",
        href: "https://www.planetkidsplaycentre.com.au/",
        image: "/images/facilities/planetkids.jpg",
        alt: "PlanetKids Play Centre",
      },
      {
        name: "Play Zone",
        href: "#",
        image: "/images/facilities/play-zone.jpg",
        alt: "Indoor Play Zone",
      },
      {
        name: "Kids Quarter",
        href: "#",
        image: "/images/facilities/kids-quarter.jpg",
        alt: "Kids Quarter",
      },
    ],
  },
];
