"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import { BookOpen, ImagePlus, Save, X, Plus } from "lucide-react";

function getAuthHeaders() {
  const user = JSON.parse(localStorage.getItem("user") || "{}");
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

export default function EditUseCasePage() {
  const router = useRouter();
  const { locale, id } = useParams() as { locale: string; id: string };

  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [categoryId, setCategoryId] = useState("");
  const [categories, setCategories] = useState<any[]>([]);

  const [tags, setTags] = useState<string[]>([]);
  const [newTag, setNewTag] = useState("");

  const [existingImgUrl, setExistingImgUrl] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageUploading, setImageUploading] = useState(false);
  const imageUploadPromiseRef = useRef<Promise<string | null> | null>(null);

  const [hasExistingNotebook, setHasExistingNotebook] = useState(false);
  const [notebookFile, setNotebookFile] = useState<File | null>(null);
  const [notebookContent, setNotebookContent] = useState<string | undefined>(undefined);

  const [fetchLoading, setFetchLoading] = useState(true);
  const [fetchError, setFetchError] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);
  const notebookInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!id) return;

    const headers = getAuthHeaders();

    Promise.all([
      fetch(`/api/usecases/${id}`, { headers }).then((r) => r.json()),
      fetch(`/api/usecases/${id}/tags`, { headers }).then((r) => r.json()),
      fetch("/api/categories", { headers }).then((r) => r.json()),
    ])
      .then(([ucJson, tagsJson, catJson]) => {
        if (!ucJson.success) {
          setFetchError(ucJson.message || ucJson.error || "Use case not found.");
          return;
        }

        const uc = ucJson.data;

        setTitle(uc.title || "");
        setDescription(uc.description || "");
        setCategoryId(uc.category_id ? String(uc.category_id) : "");

        setExistingImgUrl(uc.cover_img || null);
        setImagePreview(uc.cover_img || null);

        setHasExistingNotebook(!!uc.content);

        if (tagsJson.success && Array.isArray(tagsJson.data)) {
          setTags(tagsJson.data.map((t: any) => t.name));
        }

        if (catJson.success) {
          setCategories(catJson.data || []);
        }
      })
      .catch(() => setFetchError("Failed to load use case."))
      .finally(() => setFetchLoading(false));
  }, [id]);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));

    // Start uploading immediately in the background
    setImageUploading(true);
    const authHeaders = getAuthHeaders();
    const formData = new FormData();
    formData.append("file", file);
    formData.append("folder", "usecases");
    formData.append("bucket", "usecase-images");

    imageUploadPromiseRef.current = fetch("/api/upload", {
      method: "POST",
      headers: authHeaders,
      body: formData,
    })
      .then((r) => r.json())
      .then((json) => {
        setImageUploading(false);
        return json.success ? (json.url as string) : null;
      })
      .catch(() => {
        setImageUploading(false);
        return null;
      });
  }

  function handleNotebookChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith(".ipynb")) {
      setSaveError("Please upload a valid .ipynb notebook file.");
      return;
    }

    setSaveError("");
    setNotebookFile(file);

    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      try {
        const parsed = JSON.parse(text);
        if (!Array.isArray(parsed.cells)) throw new Error();
        setNotebookContent(text);
      } catch {
        setSaveError("Invalid notebook file — could not parse .ipynb JSON.");
        setNotebookFile(null);
        setNotebookContent(undefined);
      }
    };
    reader.readAsText(file);
  }

  function handleAddTag() {
    const cleanedTag = newTag.trim();

    if (!cleanedTag) return;

    const alreadyExists = tags.some(
      (tag) => tag.toLowerCase() === cleanedTag.toLowerCase()
    );

    if (alreadyExists) {
      setNewTag("");
      return;
    }

    setTags((prev) => [...prev, cleanedTag]);
    setNewTag("");
  }

  function handleRemoveTag(tagToRemove: string) {
    setTags((prev) => prev.filter((tag) => tag !== tagToRemove));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    setSaving(true);
    setSaveError("");

    try {
      const authHeaders = getAuthHeaders();

      let coverImgUrl: string | null = existingImgUrl;

      if (imageFile && imageUploadPromiseRef.current) {
        const uploaded = await imageUploadPromiseRef.current;
        if (!uploaded) {
          setSaveError("Image upload failed. Please remove the image and try again.");
          setSaving(false);
          return;
        }
        coverImgUrl = uploaded;
      }

      const updatedContent = notebookContent;

      const res = await fetch(`/api/usecases/${id}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders,
        },
        body: JSON.stringify({
          title,
          description,
          cover_img: coverImgUrl,
          category_id: categoryId ? Number(categoryId) : null,
          ...(updatedContent !== undefined ? { content: updatedContent } : {}),
          tags,
        }),
      });

      const json = await res.json();

      if (!json.success) {
        setSaveError(json.message || json.error || "Failed to update use case.");
        setSaving(false);
        return;
      }

      router.push(`/${locale}/admin/use-cases`);
    } catch {
      setSaveError("Failed to update use case.");
      setSaving(false);
    }
  }

  return (
    <div className="rounded-3xl border border-[#E5E7EB] bg-white p-8 shadow-sm">
      {/* Header */}
      <div className="mb-7 flex items-center gap-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-[#EAFBF0] text-[#1F8F50]">
          <BookOpen size={22} />
        </div>

        <div>
          <h2 className="text-xl font-semibold text-[#1A1A1A]">
            Edit Use Case
          </h2>
          <p className="text-sm text-[#687280]">
            Update this use case&apos;s details
          </p>
        </div>
      </div>

      {fetchLoading && (
        <p className="py-10 text-center text-sm text-[#687280]">
          Loading...
        </p>
      )}

      {fetchError && (
        <div className="rounded-xl bg-red-50 px-4 py-3 text-sm text-red-600">
          {fetchError}
        </div>
      )}

      {!fetchLoading && !fetchError && (
        <form onSubmit={handleSubmit} className="space-y-6">
          {saveError && (
            <div className="rounded-xl bg-red-50 px-4 py-3 text-sm text-red-600">
              {saveError}
            </div>
          )}

          {/* Title */}
          <div>
            <label className="mb-2 block text-sm font-medium text-[#1A1A1A]">
              Title <span className="text-red-500">*</span>
            </label>

            <input
              type="text"
              required
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Enter use case title"
              className="w-full rounded-xl border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 text-sm outline-none transition focus:border-[#2DBE6C] focus:bg-white"
            />
          </div>

          {/* Category */}
          <div>
            <label className="mb-2 block text-sm font-medium text-[#1A1A1A]">
              Category
            </label>

            <select
              value={categoryId}
              onChange={(e) => setCategoryId(e.target.value)}
              className="w-full rounded-xl border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 text-sm outline-none focus:border-[#2DBE6C] focus:bg-white"
            >
              <option value="">Select a category</option>

              {categories.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.category_name}
                </option>
              ))}
            </select>
          </div>

          {/* Description */}
          <div>
            <label className="mb-2 block text-sm font-medium text-[#1A1A1A]">
              Description
            </label>

            <textarea
              rows={4}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Enter description"
              className="w-full resize-none rounded-xl border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 text-sm outline-none transition focus:border-[#2DBE6C] focus:bg-white"
            />
          </div>

          {/* Tags */}
          <div>
            <label className="mb-3 block text-sm font-medium text-[#1A1A1A]">
              Tags
            </label>

            {/* Tags shown above input */}
            {tags.length > 0 && (
              <div className="mb-3 flex flex-wrap gap-2">
                {tags.map((tag) => (
                  <span
                    key={tag}
                    className="inline-flex items-center gap-2 rounded-full bg-[#EAFBF0] px-3 py-1 text-xs font-medium text-[#1F8F50]"
                  >
                    {tag}

                    <button
                      type="button"
                      onClick={() => handleRemoveTag(tag)}
                      className="text-[#1F8F50] transition hover:text-red-500"
                    >
                      <X size={13} />
                    </button>
                  </span>
                ))}
              </div>
            )}

            <div className="flex gap-2">
              <input
                type="text"
                value={newTag}
                onChange={(e) => setNewTag(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    handleAddTag();
                  }
                }}
                placeholder="Enter tag name"
                className="w-full rounded-xl border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 text-sm outline-none transition focus:border-[#2DBE6C] focus:bg-white"
              />

              <button
                type="button"
                onClick={handleAddTag}
                className="inline-flex items-center gap-2 rounded-xl bg-[#2DBE6C] px-5 py-3 text-sm font-medium text-white transition hover:bg-[#1F8F50]"
              >
                <Plus size={17} />
                Add
              </button>
            </div>

            {tags.length === 0 && (
              <p className="mt-2 text-xs text-[#687280]">
                No tags added yet.
              </p>
            )}
          </div>

          {/* Cover Image */}
          <div>
            <label className="mb-3 block text-sm font-medium text-[#1A1A1A]">
              Cover Image
            </label>

            <div
              onClick={() => fileInputRef.current?.click()}
              className="cursor-pointer rounded-2xl border-2 border-dashed border-[#CFEFD9] bg-[#F8FFFA] p-8 text-center transition hover:bg-[#F0FFF6]"
            >
              {imagePreview ? (
                <img
                  src={imagePreview}
                  alt="Preview"
                  className="mx-auto h-40 rounded-lg object-cover"
                />
              ) : (
                <>
                  <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-[#EAFBF0] text-[#1F8F50]">
                    <ImagePlus size={22} />
                  </div>

                  <h3 className="mt-3 text-sm font-semibold text-[#1A1A1A]">
                    Upload cover image
                  </h3>

                  <p className="mt-1 text-sm text-[#687280]">
                    Click to browse · JPEG, PNG, WebP · max 5 MB
                  </p>
                </>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileChange}
            />

            {imageUploading && (
              <p className="mt-2 text-xs text-[#687280]">Uploading image...</p>
            )}
            {imagePreview && (
              <button
                type="button"
                onClick={() => {
                  setImageFile(null);
                  setImagePreview(null);
                  setExistingImgUrl(null);
                  setImageUploading(false);
                  imageUploadPromiseRef.current = null;

                  if (fileInputRef.current) {
                    fileInputRef.current.value = "";
                  }
                }}
                className="mt-2 text-xs text-red-500 hover:underline"
              >
                Remove image
              </button>
            )}
          </div>

          {/* Python Notebook Upload */}
          <div>
            <label className="mb-3 block text-sm font-medium text-[#1A1A1A]">
              Python Notebook File
            </label>

            <div
              onClick={() => notebookInputRef.current?.click()}
              className="cursor-pointer rounded-2xl border-2 border-dashed border-[#CFEFD9] bg-[#F8FFFA] p-8 text-center transition hover:bg-[#F0FFF6]"
            >
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-[#EAFBF0] text-[#1F8F50]">
                <BookOpen size={22} />
              </div>

              <h3 className="mt-3 text-sm font-semibold text-[#1A1A1A]">
                {notebookFile
                  ? notebookFile.name
                  : hasExistingNotebook
                  ? "Notebook already uploaded"
                  : "Upload Python notebook"}
              </h3>

              <p className="mt-1 text-sm text-[#687280]">
                Click to browse · IPYNB file
              </p>
            </div>

            <input
              ref={notebookInputRef}
              type="file"
              accept=".ipynb,application/json"
              className="hidden"
              onChange={handleNotebookChange}
            />

            {(notebookFile || hasExistingNotebook) && (
              <button
                type="button"
                onClick={() => {
                  setNotebookFile(null);
                  setNotebookContent(undefined);
                  setHasExistingNotebook(false);
                  if (notebookInputRef.current) {
                    notebookInputRef.current.value = "";
                  }
                }}
                className="mt-2 text-xs text-red-500 hover:underline"
              >
                Remove notebook
              </button>
            )}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3 pt-2">
            <button
              type="submit"
              disabled={saving}
              className="inline-flex items-center gap-2 rounded-full bg-[#2DBE6C] px-6 py-3 text-sm font-medium text-white transition hover:bg-[#1F8F50] disabled:opacity-60"
            >
              <Save size={18} />
              {imageUploading && saving
                ? "Uploading image..."
                : saving
                ? "Saving..."
                : "Update Use Case"}
            </button>

            <button
              type="button"
              onClick={() => router.push(`/${locale}/admin/use-cases`)}
              className="rounded-full border border-[#E5E7EB] px-6 py-3 text-sm font-medium text-[#687280] transition hover:bg-[#F9FAFB]"
            >
              Cancel
            </button>
          </div>
        </form>
      )}
    </div>
  );
}