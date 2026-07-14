"use client";

import { useParams, useRouter } from "next/navigation";
import { useState } from "react";
import BlogForm from "../components/BlogsForm";
import AdminToast from "@/components/admin/AdminToast";
export default function AddBlog() {
  const router = useRouter();
  const { locale } = useParams<{ locale: string }>();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [toast, setToast] = useState<{ message: string; type: "success" | "error" } | null>(null);

  const handleSubmit = async (data: any) => {
    setSubmitting(true);
    setError("");

    try {
      const user = JSON.parse(localStorage.getItem("user") || "{}");
      const userId = user.userId ?? user.id ?? "";
      const roleId = user.roleId ?? user.role_id ?? "";
      const token = user.token ?? "";

      const formData = new FormData();
      formData.append("title", data.title);
      formData.append("description", data.description ?? "");
      formData.append("published_date", data.date ?? "");
      formData.append("content", data.content);
      if (data.coverImage instanceof File) {
        formData.append("cover_img", data.coverImage);
      }

      const res = await fetch("/api/blogs", {
        method: "POST",
        headers: {
          "x-user-id": String(userId),
          "x-user-role-id": String(roleId),
          "x-user-role": user.roleName ?? user.role_name ?? "",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: formData,
      });

      const json = await res.json();

      if (!res.ok || !json.success) {
        const msg = json.errors
          ? Object.values(json.errors).join(", ")
          : json.message || "Failed to create blog";
        setError(msg);
        return;
      }

      setToast({
        message: "Blog added successfully.",
        type: "success",
      });

      setTimeout(() => {
        router.push(`/${locale}/admin/blogs`);
      }, 1000);
    } catch (e) {
      console.error(e);
      setError("Something went wrong. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-[40px] font-semibold text-[#2DBE6C]">Add Blog</h1>
        <p className="mt-2 text-[16px] text-[#687280]">Create a new blog post</p>
      </div>

      <div className="rounded-2xl bg-white p-8">
        {error && (
          <div className="mb-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
            {error}
          </div>
        )}
        <BlogForm onSubmit={handleSubmit} submitting={submitting} />
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
