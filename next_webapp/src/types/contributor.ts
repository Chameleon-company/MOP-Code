// ./types/contributor.ts

export type contributor_type = "student" | "mentor" | "company_director";

export type TeamName = 
"Data Science Team" | "Website Development Team" | "Design Team" | "Cyber Security Team";

export type ContributorLevel = "Junior" | "Senior";

export interface ContributorRecord {
  _id: string;
  name: string;
  year: number;
  trimester: 1 | 2 | 3;
  contributorType: contributor_type;
  team: TeamName | null;
  position: string | null;
  level: ContributorLevel | null;
  display_order: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}
