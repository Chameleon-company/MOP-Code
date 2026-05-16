"use client";

import { useState, useEffect, useCallback } from "react";
import { ArrowUpDown } from "lucide-react";
import Pagination from "@/components/Pagination";

interface ActivityEntry {
  id: number;
  activity: string;
  performedBy: string;
  performedAt: string;
}

function authHeaders(): Record<string, string> {
  const user = JSON.parse(localStorage.getItem("user") || "{}");
  const token: string = user.token ?? "";
  return {
    "x-user-id": String(user.userId ?? user.id ?? ""),
    "x-user-role-id": String(user.roleId ?? user.role_id ?? ""),
    "x-user-role": user.roleName ?? user.role_name ?? "",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

const PAGE_SIZE = 20;

export default function ActivityHistoryPage() {
  const [entries, setEntries] = useState<ActivityEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);

  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [sortOrder, setSortOrder] = useState<"desc" | "asc">("desc");

  useEffect(() => {
    const timer = setTimeout(() => {
      setSearch(searchInput);
      setPage(1);
    }, 350);
    return () => clearTimeout(timer);
  }, [searchInput]);

  const fetchActivity = useCallback(async (
    currentPage: number,
    searchTerm: string,
    order: "desc" | "asc",
  ) => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        page: String(currentPage),
        pageSize: String(PAGE_SIZE),
        sortBy: "timestamp",
        sortOrder: order,
      });
      if (searchTerm) params.set("search", searchTerm);

      const res = await fetch(`/api/admin/activity-history?${params}`, { headers: authHeaders() });
      const json = await res.json();
      if (!json.success) throw new Error(json.message || "Failed to fetch activity");
      setEntries(json.data || []);
      setTotal(json.pagination.total);
      setTotalPages(json.pagination.totalPages);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch activity");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchActivity(page, search, sortOrder);
  }, [page, search, sortOrder, fetchActivity]);

  function toggleSort() {
    setSortOrder((prev) => (prev === "desc" ? "asc" : "desc"));
    setPage(1);
  }

  const formatTime = (iso: string) =>
    new Date(iso).toLocaleString("en-AU", {
      day: "2-digit", month: "short", year: "numeric",
      hour: "2-digit", minute: "2-digit", hour12: true,
    });

  return (
    <div className="ml-[88px] md:ml-[148px] lg:ml-0">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-[40px] font-semibold leading-[48px] text-[#2DBE6C]">Activity History</h1>
        <p className="mt-2 text-[16px] leading-[24px] text-[#687280]">
          Track all admin activities and changes
        </p>
      </div>

      {/* Search + Sort */}
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <input
          type="text"
          placeholder="Search by user name..."
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          className="rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-400"
        />

        <button
          onClick={toggleSort}
          className="flex items-center gap-1.5 rounded-lg border border-gray-300 px-3 py-2 text-sm text-[#687280] hover:border-green-400 hover:text-[#2DBE6C] transition-colors"
          title={sortOrder === "desc" ? "Showing newest first" : "Showing oldest first"}
        >
          <ArrowUpDown size={15} />
          {sortOrder === "desc" ? "Newest first" : "Oldest first"}
        </button>

        {!loading && total > 0 && (
          <span className="text-sm text-[#687280]">{total} activit{total !== 1 ? "ies" : "y"}</span>
        )}
      </div>

      {/* Table */}
      <div className="rounded-2xl bg-[#ECEAEA] p-5">
        {loading ? (
          <div className="py-16 text-center text-sm text-[#687280]">Loading...</div>
        ) : error ? (
          <div className="py-16 text-center text-sm text-red-500">{error}</div>
        ) : entries.length === 0 ? (
          <div className="py-16 text-center text-sm text-[#687280]">No activity found.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-black/30">
                  <th className="min-w-[280px] px-3 py-4 text-left text-[14px] font-semibold text-black">
                    Activity Performed
                  </th>
                  <th className="px-3 py-4 text-left text-[14px] font-semibold text-black">
                    Performed By
                  </th>
                  <th className="px-3 py-4 text-left text-[14px] font-semibold text-black">
                    Performed At
                  </th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => (
                  <tr key={entry.id} className="border-b border-black/10">
                    <td className="min-w-[280px] px-3 py-4 text-[14px] font-medium text-black">
                      {entry.activity}
                    </td>
                    <td className="px-3 py-4 text-[14px] text-[#687280]">{entry.performedBy}</td>
                    <td className="px-3 py-4 text-[14px] text-[#687280]">{formatTime(entry.performedAt)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <Pagination
        page={page}
        totalPages={totalPages}
        onPageChange={setPage}
        variant="admin"
      />
    </div>
  );
}
