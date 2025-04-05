export enum CATEGORY {
   ALL = "all",
   INTERNET = "internet",
   EV = "ev",
   SECURITY = "security",
}

export enum SEARCH_MODE {
   TITLE = "title",
   CONTENT = "content",
   TERM = "term"
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
