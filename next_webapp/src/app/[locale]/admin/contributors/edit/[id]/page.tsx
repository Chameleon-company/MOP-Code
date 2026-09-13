"use client";

import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import ContributorForm, {
  ContributorFormData,
} from "../../components/ContributorForm";
import AdminToast from "@/components/admin/AdminToast";
import { apiFetch, ApiError } from "@/lib/apiFetch";

type ApiContributor = {
  id: number | string;
  name: string;
  year: number;
  trimester: number;
  contributor_type: "student" | "mentor" | "company_director";
  team: string | null;
  position: string | null;
  level: "Senior" | "Junior" | null;
  display_order: number;
  is_active: boolean;
};

export default function EditContributorPage() {
  const router = useRouter();

  const { locale, id } = useParams<{
    locale: string;
    id: string;
  }>();

  const [initialData, setInitialData] =
    useState<Partial<ContributorFormData> | null>(
      null,
    );

  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

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

  useEffect(() => {
    const fetchContributor = async () => {
      setLoading(true);
      setError("");

      try {
        const json = await apiFetch<any>(
          `/api/contributors/${id}`,
          {
            headers: getAuthHeaders(),
            cache: "no-store",
            silent: true,
          },
        );

        const contributor: ApiContributor =
          json.data ?? json.contributor;

        if (!contributor) {
          throw new Error("Contributor not found");
        }

        setInitialData({
          name: contributor.name ?? "",
          year: String(contributor.year ?? ""),
          trimester: String(
            contributor.trimester ?? "1",
          ),
          contributorType:
            contributor.contributor_type ?? "student",
          team: contributor.team ?? "",
          position: contributor.position ?? "",
          level: contributor.level ?? "",
          displayOrder: String(
            contributor.display_order ?? 0,
          ),
          isActive: Boolean(
            contributor.is_active,
          ),
        });
      } catch (error) {
        console.error(error);

        setError(
          error instanceof Error
            ? error.message
            : "Failed to load contributor.",
        );
      } finally {
        setLoading(false);
      }
    };

    if (id) {
      fetchContributor();
    }
  }, [id]);

  const handleSubmit = async (
    data: ContributorFormData,
  ) => {
    setSubmitting(true);
    setError("");

    try {
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

      await apiFetch(
        `/api/contributors/${id}`,
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            ...getAuthHeaders(),
          },
          body: JSON.stringify(payload),
          silent: true,
        },
      );

      setToast({
        message: "Contributor updated successfully.",
        type: "success",
      });

      setTimeout(() => {
        router.push(`/${locale}/admin/contributors`);
      }, 1000);
    } catch (error) {
      console.error(error);

      const body = error instanceof ApiError ? (error.body as any) : null;
      const message = body?.errors
        ? Object.values(body.errors).flat().join(", ")
        : error instanceof Error
          ? error.message
          : "Something went wrong. Please try again.";

      setError(message);
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <div className="rounded-2xl bg-white p-8 text-[#687280]">
        Loading contributor...
      </div>
    );
  }

  if (!initialData) {
    return (
      <div>
        <div className="mb-8">
          <h1 className="text-[40px] font-semibold text-[#2DBE6C]">
            Edit Contributor
          </h1>
        </div>

        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-4 text-sm text-red-600">
          {error || "Contributor not found."}
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-[40px] font-semibold text-[#2DBE6C]">
          Edit Contributor
        </h1>

        <p className="mt-2 text-[16px] text-[#687280]">
          Update the contributor information
        </p>
      </div>

      <div className="rounded-2xl bg-white p-8">
        {error && (
          <div className="mb-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
            {error}
          </div>
        )}

        <ContributorForm
          initialData={initialData}
          onSubmit={handleSubmit}
          submitting={submitting}
          submitLabel="Update Contributor"
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