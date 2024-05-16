export enum CATEGORY {
  ALL = "all",
  INTERNET = "internet",
  EV = "ev",
  SECURITY = "security",
}

export enum SEARCH_MODE {
  TITLE = "title",
  CONTENT = "content",
}

export type SearchParams = {
  searchTerm: string;
  searchMode: SEARCH_MODE;
  category: CATEGORY;
};

export type CaseStudy = {
  id: number;
  title: string;
  content: string;
  category: CATEGORY;
  filename?: string;
};
