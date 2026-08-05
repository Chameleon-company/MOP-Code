"use client";

import { useId, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, Users, GraduationCap } from "lucide-react";
import { useTranslations } from "next-intl";
import type { ContributorRecord } from "@/types/contributor";

interface GroupedMember {
  id?: string;
  name: string;
  role?: string;
  seniority?: string;
}

interface GroupedTeam {
  teamName: string;
  members: GroupedMember[];
}

interface GroupedTrimester {
  trimester: number;
  companyDirector?: string;
  mentors: string[];
  teams: GroupedTeam[];
}

interface GroupedYear {
  year: number;
  trimesters: GroupedTrimester[];
}

function groupContributors(records: ContributorRecord[]): GroupedYear[] {
  const visible = (records ?? []).filter((record) => record.status);

  const byYear = new Map<number, Map<number, ContributorRecord[]>>();
  for (const record of visible) {
    if (!byYear.has(record.year)) byYear.set(record.year, new Map());
    const byTrimester = byYear.get(record.year)!;
    if (!byTrimester.has(record.trimester)) byTrimester.set(record.trimester, []);
    byTrimester.get(record.trimester)!.push(record);
  }

  const years: GroupedYear[] = [];

  for (const [year, byTrimester] of byYear) {
    const trimesters: GroupedTrimester[] = [];

    for (const [trimester, recordsInTrimester] of byTrimester) {
      const sorted = [...recordsInTrimester].sort(
        (a, b) => a.displayOrder - b.displayOrder
      );

      const companyDirector = sorted.find(
        (record) => record.contributorType === "Company Director"
      )?.fullName;

      const mentors = sorted
        .filter((record) => record.contributorType === "Mentor")
        .map((record) => record.fullName);

      const students = sorted.filter(
        (record) => record.contributorType === "Student"
      );

      const teamsByName = new Map<string, GroupedMember[]>();
      for (const student of students) {
        const teamName = student.team ?? "Unassigned";
        if (!teamsByName.has(teamName)) teamsByName.set(teamName, []);
        teamsByName.get(teamName)!.push({
          id: student.id,
          name: student.fullName,
          role: student.positionOrRole,
          seniority: student.level,
        });
      }

      trimesters.push({
        trimester,
        companyDirector,
        mentors,
        teams: Array.from(teamsByName, ([teamName, members]) => ({
          teamName,
          members,
        })),
      });
    }

    trimesters.sort((a, b) => a.trimester - b.trimester);
    years.push({ year, trimesters });
  }

  years.sort((a, b) => b.year - a.year);
  return years;
}

function TeamAccordion({ team }: { team: GroupedTeam }) {
  const [isOpen, setIsOpen] = useState(false);
  const panelId = useId();
  const triggerId = useId();

  return (
    <div
      className={[
        "rounded-xl border overflow-hidden bg-white dark:bg-[#263238]",
        "transition-all duration-300",
        isOpen
          ? "border-emerald-400 dark:border-emerald-500"
          : "border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-emerald-500/50",
      ].join(" ")}
    >
      <button
        id={triggerId}
        aria-controls={panelId}
        aria-expanded={isOpen}
        onClick={() => setIsOpen((prev) => !prev)}
        className="w-full flex items-center justify-between gap-3 px-4 py-3 text-left group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-inset"
      >
        <span
          className={[
            "text-sm font-semibold leading-snug transition-colors duration-200",
            isOpen
              ? "text-emerald-600 dark:text-emerald-400"
              : "text-gray-800 dark:text-gray-100 group-hover:text-emerald-600 dark:group-hover:text-emerald-400",
          ].join(" ")}
        >
          {team.teamName}
        </span>

        <span className="flex items-center gap-2 shrink-0">
          <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-600 dark:text-emerald-400 bg-emerald-100 dark:bg-emerald-500/15 px-2 py-0.5 rounded-full">
            <Users size={12} aria-hidden="true" />
            {team.members.length}
          </span>
          <motion.span
            aria-hidden="true"
            animate={{ rotate: isOpen ? 180 : 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            className={
              isOpen
                ? "text-emerald-500"
                : "text-gray-400 dark:text-gray-500 group-hover:text-emerald-500"
            }
          >
            <ChevronDown size={16} />
          </motion.span>
        </span>
      </button>

      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            id={panelId}
            role="region"
            aria-labelledby={triggerId}
            key="members"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
            className="overflow-hidden"
          >
            <ul className="flex flex-col gap-1.5 px-4 pb-3 pt-1 border-t border-gray-100 dark:border-gray-700">
              {team.members.map((member, index) => (
                <li
                  key={member.id ?? `${member.name}-${index}`}
                  className="flex items-center justify-between gap-3 text-xs sm:text-sm pt-1.5"
                >
                  <span className="text-gray-700 dark:text-gray-200">{member.name}</span>
                  <span className="text-gray-400 dark:text-gray-500">
                    {member.role ?? member.seniority ?? ""}
                  </span>
                </li>
              ))}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function TrimesterCard({ trimester }: { trimester: GroupedTrimester }) {
  const t = useTranslations("about");

  return (
    <div className="flex-1 min-w-[280px] rounded-2xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-[#1f2a30] p-5 sm:p-6 shadow-md">
      <div className="flex items-center gap-2 mb-1">
        <h3 className="text-lg font-bold text-gray-900 dark:text-white">
          {t("contributors.trimester", { number: trimester.trimester })}
        </h3>
      </div>

      {trimester.companyDirector && (
        <p className="text-xs sm:text-sm text-gray-500 dark:text-gray-400 mb-1">
          {t("contributors.companyDirector", { name: trimester.companyDirector })}
        </p>
      )}

      {trimester.mentors.length > 0 && (
        <p className="flex items-center gap-1.5 text-xs sm:text-sm text-gray-500 dark:text-gray-400 mb-4">
          <GraduationCap size={14} aria-hidden="true" />
          {t("contributors.mentoredBy", { names: trimester.mentors.join(", ") })}
        </p>
      )}

      <div className="flex flex-col gap-3">
        {trimester.teams.map((team) => (
          <TeamAccordion key={team.teamName} team={team} />
        ))}
      </div>
    </div>
  );
}

interface ContributorsSectionProps {
  contributors: ContributorRecord[];
}

export default function ContributorsSection({ contributors }: ContributorsSectionProps) {
  const t = useTranslations("about");
  const years = useMemo(() => groupContributors(contributors), [contributors]);
  const [selectedYear, setSelectedYear] = useState<number | undefined>(
    years[0]?.year
  );

  const activeYear = years.find((entry) => entry.year === selectedYear);

  return (
    <section className="w-full bg-gray-50 dark:bg-[#1f2a30] py-12 sm:py-20 px-4 sm:px-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-10">
          <span className="inline-block mb-4 px-3 py-1 rounded-full text-xs font-semibold tracking-widest uppercase bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400">
            {t("contributors.badge")}
          </span>
          <h2 className="text-2xl sm:text-4xl font-bold tracking-tight text-gray-900 dark:text-white mb-3">
            {t("contributors.title")}
          </h2>
          <p className="text-gray-500 dark:text-gray-400 text-sm sm:text-base max-w-xl mx-auto leading-relaxed">
            {t("contributors.subtitle")}
          </p>
        </div>

        {years.length === 0 ? (
          <div className="flex min-h-[200px] items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-white px-6 py-12 text-center dark:border-gray-700 dark:bg-[#263238]">
            <p className="text-base font-medium text-gray-500 dark:text-gray-400">
              {t("contributors.empty")}
            </p>
          </div>
        ) : (
          <>
            <div className="flex flex-wrap justify-center gap-2 mb-10">
              {years.map(({ year }) => {
                const isActive = year === selectedYear;
                return (
                  <button
                    key={year}
                    type="button"
                    aria-pressed={isActive}
                    onClick={() => setSelectedYear(year)}
                    className={[
                      "px-5 py-2 rounded-full text-sm font-semibold border transition-all duration-200",
                      isActive
                        ? "bg-emerald-500 border-emerald-500 text-white shadow-md"
                        : "bg-white dark:bg-[#263238] border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:border-emerald-400 hover:text-emerald-600 dark:hover:text-emerald-400",
                    ].join(" ")}
                  >
                    {year}
                  </button>
                );
              })}
            </div>

            {activeYear && (
              <div className="flex flex-wrap gap-6 items-start">
                {activeYear.trimesters.map((trimester) => (
                  <TrimesterCard key={trimester.trimester} trimester={trimester} />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
