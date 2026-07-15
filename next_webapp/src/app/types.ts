export enum CATEGORY {
  ALL = "all",
  INTERNET = "internet",
  EV = "ev",
  SECURITY = "security",
}

export enum SEARCH_MODE {
  TITLE = "title",
  CONTENT = "content",
  TAG = "tag"
}

export type SearchParams = {
  searchTerm: string;
  searchMode: SEARCH_MODE;
  category: CATEGORY;
};


export interface CaseStudy {
  id: number;
  title: string;
  description: string;
  tags: string[];
  htmlFile: string;
  category?: CATEGORY | string;
  image?: string;
}