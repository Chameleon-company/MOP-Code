export type ContributorType = "Student" | "Mentor" | "Company Director";

export type ContributorLevel = "Senior" | "Junior";

export interface ContributorRecord {
  id?: string;
  fullName: string;
  year: number;
  trimester: number;
  contributorType: ContributorType;
  team?: string;
  positionOrRole?: string;
  level?: ContributorLevel;
  displayOrder: number;
  status: boolean;
}
