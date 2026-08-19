import dbConnect from "@/lib/dbConnect";
import Contributor from "@/models/mongoose/Contributor";
import { toContributorDTO } from "@/app/api/library/contributorDto";
import type { ContributorRecord, ContributorType, ContributorLevel } from "@/types/contributor";

interface RawContributor {
  id: string;
  name: string;
  year: number;
  trimester: number;
  contributor_type: "student" | "mentor" | "company_director";
  team: string | null;
  position: string | null;
  level: "Junior" | "Senior" | null;
  display_order: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

const CONTRIBUTOR_TYPE_MAP: Record<RawContributor["contributor_type"], ContributorType> = {
  student: "Student",
  mentor: "Mentor",
  company_director: "Company Director",
};

function mapToContributorRecord(raw: RawContributor): ContributorRecord {
  return {
    id: raw.id,
    fullName: raw.name,
    year: raw.year,
    trimester: raw.trimester,
    contributorType: CONTRIBUTOR_TYPE_MAP[raw.contributor_type],
    team: raw.team ?? undefined,
    positionOrRole: raw.position ?? undefined,
    level: (raw.level ?? undefined) as ContributorLevel | undefined,
    displayOrder: raw.display_order,
    status: raw.is_active,
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
