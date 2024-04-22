export enum CATEGORY {
  ALL = "all",
  INTERNET = "internet",
  EV = "ev",
  SECURITY = "security",
}

export type CaseStudy = {
  id: number;
  title: string;
  content: string;
  category: CATEGORY;
  caseUrl?: string;
};

export const caseStudies: CaseStudy[] = [
  {
    id: 1,
    title: "Case Study 1",
    content: "Content for Case Study 1...",
    category: CATEGORY.INTERNET,
    caseUrl: "/api?filename=Childcare_Facilities_Analysis",
  },
  {
    id: 2,
    title: "Case Study 2",
    content: "Content for Case Study 2...",
    category: CATEGORY.INTERNET,
    caseUrl: "/api?filename=Projected_Music_venue_growth",
  },
  // Existing case studies
  {
    id: 3,
    title: "Case Study 3",
    content: "Content for Case Study 3...",
    category: CATEGORY.INTERNET,
  },
  {
    id: 4,
    title: "Case Study 4",
    content: "Content for Case Study 4...",
    category: CATEGORY.INTERNET,
  },
  {
    id: 5,
    title: "Case Study 5",
    content: "Content for Case Study 5...",
    category: CATEGORY.INTERNET,
  },
  {
    id: 6,
    title: "Case Study 6",
    content: "Content for Case Study 6...",
    category: CATEGORY.EV,
  },
  {
    id: 7,
    title: "Case Study 7",
    content: "Content for Case Study 7...",
    category: CATEGORY.EV,
  },
  {
    id: 8,
    title: "Case Study 8",
    content: "Content for Case Study 8...",
    category: CATEGORY.EV,
  },
  {
    id: 9,
    title: "Case Study 9",
    content: "Content for Case Study 9...",
    category: CATEGORY.EV,
  },
  {
    id: 10,
    title: "Case Study 10",
    content: "Content for Case Study 10...",
    category: CATEGORY.EV,
  },
  {
    id: 11,
    title: "Case Study 11",
    content: "Content for Case Study 11...",
    category: CATEGORY.SECURITY,
  },
  {
    id: 12,
    title: "Case Study 12",
    content: "Content for Case Study 12...",
    category: CATEGORY.SECURITY,
  },
  {
    id: 13,
    title: "Case Study 13",
    content: "Content for Case Study 13...",
    category: CATEGORY.SECURITY,
  },
  {
    id: 14,
    title: "Case Study 14",
    content: "Content for Case Study 14...",
    category: CATEGORY.SECURITY,
  },
  {
    id: 15,
    title: "Case Study 15",
    content: "Content for Case Study 15...",
    category: CATEGORY.SECURITY,
  },
  {
    id: 16,
    title: "Case Study 16",
    content: "Content for Case Study 16...",
    category: CATEGORY.SECURITY,
  },
  {
    id: 17,
    title: "Case Study 17",
    content: "Content for Case Study 17...",
    category: CATEGORY.SECURITY,
  },
  // ...and so on until you reach the desired number of case studies
];
