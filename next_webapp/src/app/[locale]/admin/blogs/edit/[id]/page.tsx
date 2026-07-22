"use client";

import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import BlogForm from "../../components/BlogsForm";

function authHeaders(user: any) {
  const userId = user.userId ?? user.id ?? "";
  const roleId = user.roleId ?? user.role_id ?? "";
  const token = user.token ?? "";
  return {
    "x-user-id": String(userId),
    "x-user-role-id": String(roleId),
    "x-user-role": user.roleName ?? user.role_name ?? "",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

export default function EditBlogPage() {
  const router = useRouter();
  const { locale, id } = useParams<{ locale: string; id: string }>();

  const [initialData, setInitialData] = useState<any>(null);
  const [loadError, setLoadError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");

  useEffect(() => {
    const fetchBlog = async () => {
      try {
        const user = JSON.parse(localStorage.getItem("user") || "{}");
        const res = await fetch(`/api/blogs/${id}`, { headers: authHeaders(user) });
        const json = await res.json();

        if (!res.ok || !json.success) {
          setLoadError(json.message || "Failed to load blog");
          return;
        }

        const b = json.data;
        setInitialData({
          title: b.title ?? "",
          description: b.description ?? "",
          date: b.published_date ?? "",
          content: b.content ?? "",
          coverImage: b.cover_img ?? "",
        });
      } catch (e) {
        console.error(e);
        setLoadError("Failed to load blog");
      }
    };

    if (id) fetchBlog();
  }, [id]);

  const handleSubmit = async (data: any) => {
    setSubmitting(true);
    setSubmitError("");

    try {
      const user = JSON.parse(localStorage.getItem("user") || "{}");

      const formData = new FormData();
      formData.append("title", data.title);
      formData.append("description", data.description ?? "");
      formData.append("published_date", data.date ?? "");
      formData.append("content", data.content);
      if (data.coverImage instanceof File) {
        formData.append("cover_img", data.coverImage);
      }

      const res = await fetch(`/api/blogs/${id}`, {
        method: "PUT",
        headers: authHeaders(user),
        body: formData,
      });

      const json = await res.json();

      if (!res.ok || !json.success) {
        const msg = json.errors
          ? Object.values(json.errors).join(", ")
          : json.message || "Failed to update blog";
        setSubmitError(msg);
        return;
      }

      router.push(`/${locale}/admin/blogs`);
    } catch (e) {
      console.error(e);
      setSubmitError("Something went wrong. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-[40px] font-semibold text-[#2DBE6C]">Edit Blog</h1>
        <p className="mt-2 text-[16px] text-[#687280]">Update blog details</p>
      </div>

      <div className="rounded-2xl bg-white p-8">
        {loadError && (
          <div className="mb-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
            {loadError}
          </div>
        )}

        {!initialData && !loadError && (
          <div className="flex justify-center py-16">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-green-500 border-t-transparent" />
          </div>
        )}

        {initialData && (
          <>
            {submitError && (
              <div className="mb-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                {submitError}
              </div>
            )}
            <BlogForm
              initialData={initialData}
              onSubmit={handleSubmit}
              submitting={submitting}
            />
          </>
        )}
      </div>
    </div>
  );
}
