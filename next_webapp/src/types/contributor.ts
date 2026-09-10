// ./types/contributor.ts

export const TEAM_ROLES = {
  "Data Science Team": [
    "Data Scientist",
    "Data Science Team Lead",
    "Data Science Assistant Team Lead",
    "Data Science Quality Manager",
  ],
  "Website Development Team": [
    "Web Developer",
    "Web Dev Team Lead",
    "Web Dev Quality Manager",
  ],
  "Design Team": [
    "Design Team Member",
    "Design Team Lead",
  ],
  "Cyber Security Team": [
    "Cyber Security Team Member",
    "Cyber Security Team Lead",
  ],
  "Project Team": [
    "Documentation Manager",
  ],
} as const;

export const TEAMS = [
  "Data Science Team",
  "Website Development Team",
  "Design Team",
  "Cyber Security Team",
  "Project Team",
] as const;

export const ROLES = [
  "Web Developer",
  "Data Scientist",
  "Data Science Team Lead",
  "Data Science Assistant Team Lead",
  "Data Science Quality Manager",
  "Web Dev Team Lead",
  "Web Dev Quality Manager",
  "Design Team Member",
  "Design Team Lead",
  "Cyber Security Team Member",
  "Cyber Security Team Lead",
  "Documentation Manager",
] as const;

export const LEVELS = ["Junior", "Senior"] as const;

export const CONTRIBUTOR_TYPES = ["student", "mentor", "company_director", "project_lead"] as const;

export type ContributorType = typeof CONTRIBUTOR_TYPES[number];

export type TeamName = keyof typeof TEAM_ROLES;

export type ContributorLevel = typeof LEVELS[number];

export interface ContributorRecord {
  _id: string;
  name: string;
  year: number;
  trimester: 1 | 2 | 3;
  contributor_type: ContributorType;
  team: TeamName | null;
  position: string | null;
  level: ContributorLevel | null;
  display_order: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}
