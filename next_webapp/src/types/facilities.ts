export type FacilityItem = {
  name: string;
  href: string;
  image: string; // path under /public (e.g., /images/facilities/mcg.jpg)
  alt: string;
};

export type FacilitySectionData = {
  id: string;
  title: string;
  subtitle: string;
  items: FacilityItem[];
  /** choose the panel color for the section */
  tone?: "green" | "blue";
};
