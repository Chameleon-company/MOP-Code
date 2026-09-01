"use client";

import {
  ChevronDown,
  Edit,
  Plus,
  Search,
  Trash2,
  Users,
} from "lucide-react";
import { useParams, useRouter } from "next/navigation";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import AdminToast from "@/components/admin/AdminToast";

type CustomSelectOption = {
  value: string;
  label: string;
  icon?: string;
};

type CustomSelectProps = {
  value: string;
  onChange: (value: string) => void;
  options: CustomSelectOption[];
  placeholder?: string;
  disabled?: boolean;
  className?: string;
};

function CustomSelect({
  value,
  onChange,
  options,
  placeholder = "Select option",
  disabled = false,
  className = "",
}: CustomSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () =>
      document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const selectedOption = options.find((opt) => opt.value === value);

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={`w-full rounded-xl border border-gray-200 bg-white px-4 py-3 text-sm outline-none transition focus:border-[#2DBE6C] focus:ring-2 focus:ring-[#2DBE6C]/10 flex items-center justify-between ${
          disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer hover:border-gray-300"
        } ${className}`}
      >
        <span className={selectedOption ? "text-gray-800" : "text-gray-500"}>
          {selectedOption ? selectedOption.label : placeholder}
        </span>
        <ChevronDown
          size={16}
          className={`text-gray-400 transition ${isOpen ? "rotate-180" : ""}`}
        />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 right-0 mt-2 rounded-xl border border-gray-200 bg-white shadow-lg z-10">
          {options.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => {
                onChange(option.value);
                setIsOpen(false);
              }}
              className={`w-full px-4 py-3 text-left text-sm transition first:rounded-t-xl last:rounded-b-xl ${
                value === option.value
                  ? "bg-[#2DBE6C]/10 text-[#2DBE6C] font-medium"
                  : "text-gray-800 hover:bg-gray-50"
              }`}
            >
              <span>
                {option.icon && <span className="mr-2">{option.icon}</span>}
                {option.label}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

type Contributor = {
  id: number | string;
  name: string;
  year: number;
  trimester: number;
  contributor_type: "student" | "mentor" | "company_director";
  team: string | null;
  position: string | null;
  display_order: number;
  is_active: boolean;
};

export default function ContributorsPage() {
  const router = useRouter();
  const { locale } = useParams<{ locale: string }>();

  const [contributors, setContributors] = useState<
    Contributor[]
  >([
    {
      id: 1,
      name: "Alice Johnson",
      year: 2026,
      trimester: 1,
      contributor_type: "student",
      team: "Data Science Team",
      position: "Project Leader",
      display_order: 1,
      is_active: true,
    },
    {
      id: 2,
      name: "Bob Smith",
      year: 2026,
      trimester: 1,
      contributor_type: "student",
      team: "Development Team",
      position: "Team Leader",
      display_order: 2,
      is_active: true,
    },
    {
      id: 3,
      name: "Dr. Sarah Wilson",
      year: 2026,
      trimester: 1,
      contributor_type: "mentor",
      team: null,
      position: "Project Mentor",
      display_order: 3,
      is_active: true,
    },
    {
      id: 4,
      name: "Carol Davis",
      year: 2026,
      trimester: 1,
      contributor_type: "student",
      team: "Data Science Team",
      position: "Assistant Team Leader",
      display_order: 4,
      is_active: true,
    },
    {
      id: 5,
      name: "John Brown",
      year: 2026,
      trimester: 1,
      contributor_type: "company_director",
      team: null,
      position: null,
      display_order: 5,
      is_active: true,
    },
    {
      id: 6,
      name: "Emily Chen",
      year: 2026,
      trimester: 1,
      contributor_type: "student",
      team: "Development Team",
      position: "Co-Leader",
      display_order: 6,
      is_active: false,
    },
  ]);

  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<
    string | number | null
  >(null);

  const [deleteConfirm, setDeleteConfirm] = useState<{
    contributor: Contributor | null;
    isOpen: boolean;
  }>({
    contributor: null,
    isOpen: false,
  });

  const [search, setSearch] = useState("");
  const [year, setYear] = useState("");
  const [trimester, setTrimester] = useState("");
  const [type, setType] = useState("");

  const [toast, setToast] = useState<{
    message: string;
    type: "success" | "error";
  } | null>(null);

  const getAuthHeaders = () => {
    const user = JSON.parse(
      localStorage.getItem("user") || "{}",
    );

    const userId = user.userId ?? user.id ?? "";
    const roleId = user.roleId ?? user.role_id ?? "";
    const token = user.token ?? "";

    return {
      "x-user-id": String(userId),
      "x-user-role-id": String(roleId),
      "x-user-role":
        user.roleName ?? user.role_name ?? "",
      ...(token
        ? {
            Authorization: `Bearer ${token}`,
          }
        : {}),
    };
  };

  const fetchContributors = useCallback(async () => {
    setLoading(true);

    try {
      const response = await fetch("/api/contributors", {
        headers: getAuthHeaders(),
        cache: "no-store",
      });

      const json = await response.json();

      if (!response.ok || !json.success) {
        throw new Error(
          json.message || "Failed to load contributors",
        );
      }

      const contributorData =
        json.data ?? json.contributors ?? [];

      setContributors(
        Array.isArray(contributorData)
          ? contributorData
          : [],
      );
    } catch (error) {
      console.error(error);

      setToast({
        message: "Failed to load contributors.",
        type: "error",
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchContributors();
  }, [fetchContributors]);

  const years = useMemo(() => {
    return Array.from(
      new Set(
        contributors.map((contributor) =>
          String(contributor.year),
        ),
      ),
    ).sort((a, b) => Number(b) - Number(a));
  }, [contributors]);

  const filteredContributors = useMemo(() => {
    const normalizedSearch = search
      .trim()
      .toLowerCase();

    return contributors
      .filter((contributor) => {
        const matchesSearch =
          !normalizedSearch ||
          contributor.name
            .toLowerCase()
            .includes(normalizedSearch) ||
          contributor.team
            ?.toLowerCase()
            .includes(normalizedSearch) ||
          contributor.position
            ?.toLowerCase()
            .includes(normalizedSearch);

        const matchesYear =
          !year ||
          String(contributor.year) === year;

        const matchesTrimester =
          !trimester ||
          String(contributor.trimester) === trimester;

        const matchesType =
          !type ||
          contributor.contributor_type === type;

        return (
          matchesSearch &&
          matchesYear &&
          matchesTrimester &&
          matchesType
        );
      })
      .sort((a, b) => {
        if (b.year !== a.year) {
          return b.year - a.year;
        }

        if (b.trimester !== a.trimester) {
          return b.trimester - a.trimester;
        }

        return (
          (a.display_order ?? 0) -
          (b.display_order ?? 0)
        );
      });
  }, [
    contributors,
    search,
    year,
    trimester,
    type,
  ]);

  const handleDelete = async (
    contributor: Contributor,
  ) => {
    setDeleteConfirm({
      contributor,
      isOpen: true,
    });
  };

  const confirmDelete = async () => {
    const contributor = deleteConfirm.contributor;
    if (!contributor) return;

    setDeleteConfirm({
      contributor: null,
      isOpen: false,
    });
    setDeletingId(contributor.id);

    try {
      const response = await fetch(
        `/api/contributors/${contributor.id}`,
        {
          method: "DELETE",
          headers: getAuthHeaders(),
        },
      );

      const json = await response.json();

      if (!response.ok || !json.success) {
        throw new Error(
          json.message || "Failed to delete contributor",
        );
      }

      setContributors((previous) =>
        previous.filter(
          (item) => item.id !== contributor.id,
        ),
      );

      setToast({
        message: "Contributor deleted successfully.",
        type: "success",
      });
    } catch (error) {
      console.error("Delete error:", error);

      // Fallback: remove from local state even if API fails
      // This allows testing with dummy data
      setContributors((previous) =>
        previous.filter(
          (item) => item.id !== contributor.id,
        ),
      );

      setToast({
        message:
          error instanceof Error
            ? error.message
            : "Failed to delete contributor.",
        type: "error",
      });
    } finally {
      setDeletingId(null);
    }
  };

  const cancelDelete = () => {
    setDeleteConfirm({
      contributor: null,
      isOpen: false,
    });
  };

  return (
    <div>
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold leading-tight text-[#2DBE6C] sm:text-3xl md:text-[40px]">
            Contributors
          </h1>

          <p className="mt-1 text-xs text-[#687280] sm:text-sm md:text-[16px]">
            Manage students and mentors displayed on the website
          </p>
        </div>

        <button
          type="button"
          onClick={() => router.push(`/${locale}/admin/contributors/add`)}
          className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-[#2DBE6C] px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-[#25A85E] sm:px-5 sm:py-3"
        >
          <Plus size={18} />
          Add Contributor
        </button>
      </div>

      <div className="rounded-2xl bg-white p-6">
        <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-4">
          <div className="relative">
            <Search
              size={18}
              className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400"
            />

            <input
              type="text"
              value={search}
              onChange={(event) =>
                setSearch(event.target.value)
              }
              placeholder="Search contributors"
              className="w-full rounded-xl border border-gray-200 py-3 pl-11 pr-4 text-sm outline-none focus:border-[#2DBE6C]"
            />
          </div>

          <CustomSelect
            value={year}
            onChange={setYear}
            options={[
              { value: "", label: "All years" },
              ...years.map((yearOption) => ({
                value: yearOption,
                label: yearOption,
              })),
            ]}
            placeholder="All years"
          />

          <CustomSelect
            value={trimester}
            onChange={setTrimester}
            options={[
              { value: "", label: "All trimesters" },
              { value: "1", label: "Trimester 1" },
              { value: "2", label: "Trimester 2" },
              { value: "3", label: "Trimester 3" },
            ]}
            placeholder="All trimesters"
          />

          <CustomSelect
            value={type}
            onChange={setType}
            options={[
              { value: "", label: "All types" },
              { value: "student", label: "👨‍🎓 Students" },
              { value: "mentor", label: "👨‍🏫 Mentors" },
              { value: "company_director", label: "🏢 Company Directors" },
            ]}
            placeholder="All types"
          />
        </div>

        {loading ? (
          <div className="flex min-h-[250px] items-center justify-center text-[#687280]">
            Loading contributors...
          </div>
        ) : filteredContributors.length === 0 ? (
          <div className="flex min-h-[250px] flex-col items-center justify-center text-center">
            <div className="mb-4 rounded-full bg-[#2DBE6C]/10 p-4">
              <Users
                size={32}
                className="text-[#2DBE6C]"
              />
            </div>

            <h2 className="text-lg font-semibold text-[#344054]">
              No contributors found
            </h2>

            <p className="mt-2 text-sm text-[#687280]">
              Add a contributor or change your filters.
            </p>
          </div>
        ) : (
          <>
            {/* Mobile Card View (< 768px) */}
            <div className="space-y-3 md:hidden">
              {filteredContributors.map((contributor) => (
                <div
                  key={contributor.id}
                  className="flex flex-col gap-3 rounded-xl border border-gray-200 bg-white p-4 shadow-sm"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h4 className="text-base font-semibold text-[#344054]">
                        {contributor.name}
                      </h4>
                      <p className="mt-0.5 text-xs text-[#687280]">
                        T{contributor.trimester}, {contributor.year}
                      </p>
                    </div>

                    <div className="flex shrink-0 items-center gap-2">
                      <button
                        type="button"
                        onClick={() =>
                          router.push(
                            `/${locale}/admin/contributors/edit/${contributor.id}`,
                          )
                        }
                        className="rounded-lg border border-gray-200 p-2 text-[#687280] transition hover:border-[#2DBE6C] hover:text-[#2DBE6C]"
                        aria-label={`Edit ${contributor.name}`}
                      >
                        <Edit size={16} />
                      </button>

                      <button
                        type="button"
                        onClick={() => handleDelete(contributor)}
                        disabled={deletingId === contributor.id}
                        className="rounded-lg border border-red-100 p-2 text-red-500 transition hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-50"
                        aria-label={`Delete ${contributor.name}`}
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2 border-t border-gray-100 pt-2 text-xs">
                    <div>
                      <span className="block text-[11px] text-[#687280]">
                        Type
                      </span>
                      <span className="mt-0.5 inline-block rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-medium capitalize text-blue-600">
                        {contributor.contributor_type}
                      </span>
                    </div>

                    <div>
                      <span className="block text-[11px] text-[#687280]">
                        Status
                      </span>
                      <span
                        className={`mt-0.5 inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${
                          contributor.is_active
                            ? "bg-green-50 text-green-600"
                            : "bg-gray-100 text-gray-500"
                        }`}
                      >
                        {contributor.is_active ? "Active" : "Inactive"}
                      </span>
                    </div>

                    <div>
                      <span className="block text-[11px] text-[#687280]">
                        Team
                      </span>
                      <span className="font-medium text-[#344054]">
                        {contributor.team || "—"}
                      </span>
                    </div>

                    <div>
                      <span className="block text-[11px] text-[#687280]">
                        Role
                      </span>
                      <span className="font-medium text-[#344054]">
                        {contributor.position || "—"}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Desktop Table View (>= 768px) */}
            <div className="hidden overflow-x-auto md:block">
              <table className="w-full min-w-[1000px] border-collapse">
                <thead>
                  <tr className="border-b border-gray-100 text-left">
                    <th className="px-4 py-4 text-sm font-semibold text-[#344054]">
                      Name
                    </th>

                    <th className="px-4 py-4 text-sm font-semibold text-[#344054]">
                      Period
                    </th>

                    <th className="px-4 py-4 text-sm font-semibold text-[#344054]">
                      Type
                    </th>

                    <th className="px-4 py-4 text-sm font-semibold text-[#344054]">
                      Team
                    </th>

                    <th className="px-4 py-4 text-sm font-semibold text-[#344054]">
                      Role
                    </th>

                    <th className="px-4 py-4 text-sm font-semibold text-[#344054]">
                      Status
                    </th>

                    <th className="px-4 py-4 text-right text-sm font-semibold text-[#344054]">
                      Actions
                    </th>
                  </tr>
                </thead>

                <tbody>
                  {filteredContributors.map((contributor) => (
                    <tr
                      key={contributor.id}
                      className="border-b border-gray-100 last:border-b-0"
                    >
                      <td className="px-4 py-4">
                        <p className="font-medium text-[#344054]">
                          {contributor.name}
                        </p>
                      </td>

                      <td className="px-4 py-4 text-sm text-[#687280]">
                        T{contributor.trimester}, {contributor.year}
                      </td>

                      <td className="px-4 py-4">
                        <span className="rounded-full bg-blue-50 px-3 py-1 text-xs font-medium capitalize text-blue-600">
                          {contributor.contributor_type}
                        </span>
                      </td>

                      <td className="max-w-[230px] px-4 py-4 text-sm text-[#687280]">
                        {contributor.team || "—"}
                      </td>

                      <td className="max-w-[220px] px-4 py-4 text-sm text-[#687280]">
                        {contributor.position || "—"}
                      </td>

                      <td className="px-4 py-4">
                        <span
                          className={`rounded-full px-3 py-1 text-xs font-medium ${
                            contributor.is_active
                              ? "bg-green-50 text-green-600"
                              : "bg-gray-100 text-gray-500"
                          }`}
                        >
                          {contributor.is_active ? "Active" : "Inactive"}
                        </span>
                      </td>

                      <td className="px-4 py-4">
                        <div className="flex justify-end gap-2">
                          <button
                            type="button"
                            onClick={() =>
                              router.push(
                                `/${locale}/admin/contributors/edit/${contributor.id}`,
                              )
                            }
                            className="rounded-lg border border-gray-200 p-2 text-[#687280] transition hover:border-[#2DBE6C] hover:text-[#2DBE6C]"
                            aria-label={`Edit ${contributor.name}`}
                          >
                            <Edit size={17} />
                          </button>

                          <button
                            type="button"
                            onClick={() => handleDelete(contributor)}
                            disabled={deletingId === contributor.id}
                            className="rounded-lg border border-red-100 p-2 text-red-500 transition hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-50"
                            aria-label={`Delete ${contributor.name}`}
                          >
                            <Trash2 size={17} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>

      {toast && (
        <AdminToast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}

      {deleteConfirm.isOpen && deleteConfirm.contributor && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="rounded-2xl bg-white p-8 max-w-md mx-4">
            <h2 className="text-xl font-semibold text-[#344054]">
              Delete Contributor
            </h2>

            <p className="mt-4 text-[15px] text-[#687280]">
              Are you sure you want to delete{" "}
              <span className="font-medium text-[#344054]">
                {deleteConfirm.contributor.name}
              </span>
              ? This action cannot be undone.
            </p>

            <div className="mt-8 flex justify-end gap-3">
              <button
                type="button"
                onClick={cancelDelete}
                disabled={deletingId !== null}
                className="rounded-lg border border-gray-200 px-5 py-2 text-sm font-medium text-[#687280] transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Cancel
              </button>

              <button
                type="button"
                onClick={confirmDelete}
                disabled={deletingId !== null}
                className="rounded-lg bg-red-500 px-5 py-2 text-sm font-medium text-white transition hover:bg-red-600 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {deletingId === deleteConfirm.contributor.id
                  ? "Deleting..."
                  : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}