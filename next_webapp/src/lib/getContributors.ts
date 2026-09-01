import dbConnect from "@/lib/dbConnect";
import Contributor, { TeamName } from "@/models/mongoose/Contributor";
import { toContributorDTO } from "@/app/api/library/contributorDto";
import type { ContributorRecord, ContributorType, ContributorLevel } from "@/types/contributor";

interface RawContributor {
  id: string;
  name: string;
  year: number;
  trimester: number;
  contributor_type: "student" | "mentor" | "company_director" | "project_lead";
  team: string | null;
  position: string | null;
  level: "Junior" | "Senior" | null;
  display_order: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

const CONTRIBUTOR_TYPE_MAP: Record<RawContributor["contributor_type"], ContributorType> = {
  student: "student",
  mentor: "mentor",
  company_director: "company_director",
  project_lead: "project_lead",
};

function mapToContributorRecord(raw: RawContributor): ContributorRecord {
  return {
    _id: raw.id,
    name: raw.name,
    year: raw.year,
    trimester: raw.trimester as 1 | 2 | 3, //casted b/c RawContributor.trimester is a number, but ContributorRecord.trimester is a union type of 1 | 2 | 3
    contributor_type: CONTRIBUTOR_TYPE_MAP[raw.contributor_type],
    team: (raw.team ?? null) as TeamName | null, // casted b/c RawContributor.team is a string | null, but ContributorRecord.team is a union type of TeamName | null
    position: raw.position ?? null,
    level: (raw.level ?? null) as ContributorLevel | null,
    display_order: raw.display_order,
    is_active: raw.is_active,
    created_at: raw.created_at,
    updated_at: raw.updated_at,
  };
}

// Server-only data access for the public "about" page. Reads straight from
// the same collection /api/contributors serves, without a self-referential
// HTTP round trip (that route stays in place for the admin CRUD screens).
export async function getContributors(): Promise<ContributorRecord[]> {
  try {
    await dbConnect();

    const contributors = await Contributor.find({})
      .sort({ year: -1, trimester: 1, display_order: 1 })
      .lean();

    return contributors.map((doc) =>
      mapToContributorRecord(toContributorDTO(doc) as RawContributor)
    );
  } catch (error) {
    console.error("getContributors Error:", error);
    return [];
  }
}
