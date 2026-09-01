"use client";

import { useEffect, useState } from "react";

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

function formatTime(iso: string) {
  return new Date(iso).toLocaleString("en-AU", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  });
}

export default function AdminRecentActivity() {
  const [entries, setEntries] = useState<ActivityEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/admin/activity-history?pageSize=5", { headers: authHeaders() })
      .then((r) => r.json())
      .then((json) => {
        if (json.success) setEntries(json.data || []);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="mt-8 md:mt-10">
      <h2 className="mb-4 text-2xl font-semibold leading-tight text-[#2DBE6C] sm:text-3xl md:text-[28px]">
        Recent Activity
      </h2>

      <div className="w-full rounded-2xl bg-[#ECEAEA] p-3 shadow-sm sm:p-5 md:p-6">
        {/* Mobile Card View (< 768px) */}
        <div className="space-y-3 md:hidden">
          {loading && (
            <div className="rounded-xl bg-white p-6 text-center text-xs text-[#687280]">
              Loading...
            </div>
          )}
          {!loading && entries.length === 0 && (
            <div className="rounded-xl bg-white p-6 text-center text-xs text-[#687280]">
              No recent activity.
            </div>
          )}
          {!loading &&
            entries.map((entry) => (
              <div
                key={entry.id}
                className="flex flex-col gap-2 rounded-xl border border-black/5 bg-white p-3.5 shadow-sm"
              >
                <p className="text-xs font-medium text-black sm:text-sm">
                  {entry.activity}
                </p>
                <div className="flex items-center justify-between border-t border-black/5 pt-2 text-[11px] text-[#687280]">
                  <span>
                    By:{" "}
                    <span className="font-medium text-black">
                      {entry.performedBy}
                    </span>
                  </span>
                  <span>{formatTime(entry.performedAt)}</span>
                </div>
              </div>
            ))}
        </div>

        {/* Desktop Table View (>= 768px) */}
        <div className="hidden overflow-x-auto md:block">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-black">
                <th className="px-2 py-3 text-left text-[13px] font-semibold text-black sm:text-[14px]">
                  Date
                </th>
                <th className="px-2 py-3 text-left text-[13px] font-semibold text-black sm:text-[14px]">
                  Activity
                </th>
                <th className="px-2 py-3 text-left text-[13px] font-semibold text-black sm:text-[14px]">
                  Performed By
                </th>
              </tr>
            </thead>
            <tbody>
              {loading && (
                <tr>
                  <td
                    colSpan={3}
                    className="px-2 py-6 text-center text-[13px] text-[#687280]"
                  >
                    Loading...
                  </td>
                </tr>
              )}
              {!loading && entries.length === 0 && (
                <tr>
                  <td
                    colSpan={3}
                    className="px-2 py-6 text-center text-[13px] text-[#687280]"
                  >
                    No recent activity.
                  </td>
                </tr>
              )}
              {entries.map((entry) => (
                <tr
                  key={entry.id}
                  className="border-b border-gray-300 last:border-b-0"
                >
                  <td className="px-2 py-3 text-[13px] text-black sm:text-[14px]">
                    {formatTime(entry.performedAt)}
                  </td>
                  <td className="px-2 py-3 text-[13px] text-black sm:text-[14px]">
                    {entry.activity}
                  </td>
                  <td className="px-2 py-3 text-[13px] text-black sm:text-[14px]">
                    {entry.performedBy}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
