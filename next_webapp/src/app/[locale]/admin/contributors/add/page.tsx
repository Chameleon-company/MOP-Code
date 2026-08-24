"use client";

import { useParams, useRouter } from "next/navigation";
import { useState } from "react";
import ContributorForm, {
  ContributorFormData,
} from "../components/ContributorForm";
import AdminToast from "@/components/admin/AdminToast";

export default function AddContributorPage() {
  const router = useRouter();
  const { locale } = useParams<{ locale: string }>();

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const [toast, setToast] = useState<{
    message: string;
    type: "success" | "error";
  } | null>(null);

  const handleSubmit = async (
    data: ContributorFormData,
  ) => {
    setSubmitting(true);
    setError("");

    try {
      const user = JSON.parse(
        localStorage.getItem("user") || "{}",
      );

      const userId = user.userId ?? user.id ?? "";
      const roleId = user.roleId ?? user.role_id ?? "";
      const token = user.token ?? "";

      const payload = {
        name: data.name.trim(),
        year: Number(data.year),
        trimester: Number(data.trimester),
        contributor_type: data.contributorType,
        team:
          data.contributorType === "student"
            ? data.team.trim()
            : null,
        position: data.position.trim() || null,
        level:
          data.contributorType === "student"
            ? data.level || null
            : null,
        display_order: Number(data.displayOrder) || 0,
        is_active: data.isActive,
      };

      const response = await fetch("/api/contributors", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-user-id": String(userId),
          "x-user-role-id": String(roleId),
          "x-user-role":
            user.roleName ?? user.role_name ?? "",
          ...(token
            ? {
                Authorization: `Bearer ${token}`,
              }
            : {}),
        },
        body: JSON.stringify(payload),
      });

      const json = await response.json();

      console.log("API Response:", { status: response.ok, success: json.success, data: json });

      if (!response.ok || !json.success) {
        const message = json.errors
          ? Object.values(json.errors).flat().join(", ")
          : json.message || "Failed to add contributor";

        setError(message);
        return;
      }

      setToast({
        message: "Contributor added successfully.",
        type: "success",
      });

      setTimeout(() => {
        router.push(`/${locale}/admin/contributors`);
      }, 1000);
    } catch (error) {
      console.error(error);

      setError(
        "Something went wrong. Please try again.",
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-[40px] font-semibold text-[#2DBE6C]">
          Add Contributor
        </h1>

        <p className="mt-2 text-[16px] text-[#687280]">
          Add a student or mentor to the contributors page
        </p>
      </div>

      <div className="rounded-2xl bg-white p-8">
        {error && (
          <div className="mb-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
            {error}
          </div>
        )}

        <ContributorForm
          onSubmit={handleSubmit}
          submitting={submitting}
          submitLabel="Add Contributor"
        />
      </div>

      {toast && (
        <AdminToast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  );
}