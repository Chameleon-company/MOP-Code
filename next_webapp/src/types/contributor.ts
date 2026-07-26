export interface ContributorMember {
  name: string;
  role?: string | null;
  seniority?: string | null;
  [key: string]: unknown;
}

export interface ContributorTeam {
  teamName: string;
  members: ContributorMember[];
}

export interface ContributorTrimester {
  trimester: number;
  companyDirector?: string | null;
  mentors: string[];
  teams: ContributorTeam[];
  note?: string;
}

export interface ContributorYear {
  year: number;
  trimesters: ContributorTrimester[];
}
