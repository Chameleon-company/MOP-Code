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
  name: string;
  description: string;
  tags: string[];
  filename?: string;
};

export type UseCase = {
  id: number;
  name: String,
  auth: String,
  duration: String,
  level: String,
  skills: String,
  description: String,
  tags: [String],
  filename?: String
}
