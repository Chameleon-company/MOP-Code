"use client";

import { useState, useEffect, useCallback, use } from "react";
import { Plus, X, Upload, Search, Pencil } from "lucide-react";
import AdminToast from "@/components/admin/AdminToast";
import ConfirmModal from "@/components/admin/ConfirmModal";

type GalleryImage = {
  id: number;
  title: string;
  img_url: string;
  created_at: string;
};

function authHeaders(): Record<string, string> {
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

const PAGE_SIZE = 12;

export default function GalleryPage({ params }: { params: Promise<{ locale: string }> }) {
  use(params);

  const [images, setImages] = useState<GalleryImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");

  const [selectedImage, setSelectedImage] = useState<GalleryImage | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<GalleryImage | null>(null);

  // Upload modal
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadTitle, setUploadTitle] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadPreview, setUploadPreview] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  // Edit modal
  const [editTarget, setEditTarget] = useState<GalleryImage | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [editFile, setEditFile] = useState<File | null>(null);
  const [editPreview, setEditPreview] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const [toast, setToast] = useState<{ message: string; type: "success" | "error" } | null>(null);

  // ── Fetch ──────────────────────────────────────────────────────────────────
  const fetchImages = useCallback(async (searchTerm: string, currentPage: number) => {
    setLoading(true);
    try {
      const qs = new URLSearchParams({
        page: String(currentPage),
        pageSize: String(PAGE_SIZE),
      });
      if (searchTerm) qs.set("search", searchTerm);

      const res = await fetch(`/api/gallery?${qs}`, { headers: authHeaders() });
      const json = await res.json();
      if (json.success) {
        setImages(json.data ?? []);
        setTotal(json.pagination?.total ?? 0);
        setTotalPages(json.pagination?.totalPages ?? 1);
      } else {
        setToast({ message: json.message || "Failed to fetch gallery.", type: "error" });
      }
    } catch {
      setToast({ message: "Failed to fetch gallery.", type: "error" });
    } finally {
      setLoading(false);
    }
  }, []);

  // Debounce: wait 400 ms after the user stops typing before fetching
  useEffect(() => {
    const timer = setTimeout(() => {
      setSearch(searchInput);
      setPage(1);
    }, 400);
    return () => clearTimeout(timer);
  }, [searchInput]);

  useEffect(() => {
    fetchImages(search, page);
  }, [search, page, fetchImages]);

  // ── Upload ─────────────────────────────────────────────────────────────────
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadFile(file);
    setUploadPreview(URL.createObjectURL(file));
  };

  const handleUpload = async () => {
    if (!uploadTitle.trim() || !uploadFile) return;
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("title", uploadTitle.trim());
      formData.append("image", uploadFile);

      const res = await fetch("/api/gallery", {
        method: "POST",
        headers: authHeaders(),
        body: formData,
      });
      const json = await res.json();

      if (json.success) {
        setToast({ message: "Gallery photo uploaded successfully.", type: "success" });
        closeUpload();
        fetchImages(search, 1);
        setPage(1);
      } else {
        const errMsg =
          json.errors?.title || json.errors?.image || json.message || "Upload failed.";
        setToast({ message: errMsg, type: "error" });
      }
    } catch {
      setToast({ message: "Upload failed.", type: "error" });
    } finally {
      setUploading(false);
    }
  };

  const closeUpload = () => {
    setUploadOpen(false);
    setUploadTitle("");
    setUploadFile(null);
    setUploadPreview(null);
  };

  // ── Edit ───────────────────────────────────────────────────────────────────
  const openEdit = (img: GalleryImage) => {
    setSelectedImage(null);
    setEditTarget(img);
    setEditTitle(img.title);
    setEditFile(null);
    setEditPreview(null);
  };

  const handleEditFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setEditFile(file);
    setEditPreview(URL.createObjectURL(file));
  };

  const hasEditChanges =
    !!editTarget &&
    (editTitle.trim() !== editTarget.title || !!editFile);

  const handleSaveEdit = async () => {
    if (!editTarget || !hasEditChanges) return;
    setSaving(true);
    try {
      const formData = new FormData();
      // Always send title so the API receives at least one field
      formData.append("title", editTitle.trim() || editTarget.title);
      if (editFile) formData.append("image", editFile);

      const res = await fetch(`/api/gallery/${editTarget.id}`, {
        method: "PUT",
        headers: authHeaders(),
        body: formData,
      });
      const json = await res.json();

      if (json.success) {
        setToast({ message: "Gallery photo updated successfully.", type: "success" });
        closeEdit();
        fetchImages(search, page);
      } else {
        const errMsg =
          json.errors?.title || json.errors?.image || json.message || "Update failed.";
        setToast({ message: errMsg, type: "error" });
      }
    } catch {
      setToast({ message: "Update failed.", type: "error" });
    } finally {
      setSaving(false);
    }
  };

  const closeEdit = () => {
    setEditTarget(null);
    setEditTitle("");
    setEditFile(null);
    setEditPreview(null);
  };

  // ── Delete ─────────────────────────────────────────────────────────────────
  const confirmDelete = async () => {
    if (!deleteTarget) return;
    try {
      const res = await fetch(`/api/gallery/${deleteTarget.id}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      const json = await res.json();

      if (json.success) {
        setToast({ message: "Gallery photo deleted successfully.", type: "success" });
        setDeleteTarget(null);
        setSelectedImage(null);
        const newPage = images.length === 1 && page > 1 ? page - 1 : page;
        setPage(newPage);
        fetchImages(search, newPage);
      } else {
        setToast({ message: json.message || "Delete failed.", type: "error" });
        setDeleteTarget(null);
      }
    } catch {
      setToast({ message: "Delete failed.", type: "error" });
      setDeleteTarget(null);
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="-m-8">
      {/* ── Page header ─────────────────────────────────────── */}
      <div className="flex flex-col gap-4 px-8 py-6 sm:flex-row sm:items-center sm:justify-between">
        <h1 className="text-[40px] font-semibold leading-[48px] text-[#2DBE6C]">Gallery</h1>

        <div className="flex items-center gap-3">
          {/* Search */}
          <div className="flex items-center gap-2 rounded-xl border border-[#CFEFD9] bg-[#F8FFFA] px-4 py-2.5">
            <Search size={16} className="text-[#1F8F50]" />
            <input
              placeholder="Search by title..."
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              className="w-44 bg-transparent text-sm outline-none"
            />
            {searchInput && (
              <button
                type="button"
                onClick={() => setSearchInput("")}
                className="text-gray-400 hover:text-gray-600"
              >
                <X size={14} />
              </button>
            )}
          </div>

          <button
            type="button"
            onClick={() => setUploadOpen(true)}
            className="inline-flex items-center gap-2 rounded-lg bg-[#1F8F50] px-5 py-3 text-[14px] font-medium text-white transition hover:bg-[#2DBE6C]"
          >
            <Plus size={16} />
            Upload New Photo
          </button>
        </div>
      </div>

      {/* ── Photo grid ──────────────────────────────────────── */}
      {loading ? (
        <div className="flex justify-center py-32">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-green-500 border-t-transparent" />
        </div>
      ) : images.length === 0 ? (
        <div className="col-span-4 flex flex-col items-center justify-center border-2 border-dashed border-gray-200 bg-white py-32 text-gray-400">
          <p className="text-[16px] font-medium">No photos found</p>
          <p className="mt-1 text-[14px]">
            {searchInput ? `No photos found for "${searchInput}"` : 'Click "Upload New Photo" to get started'}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-1 sm:grid-cols-3 lg:grid-cols-4">
          {images.map((image) => (
            <button
              key={image.id}
              type="button"
              onClick={() => setSelectedImage(image)}
              className="group relative overflow-hidden focus:outline-none"
            >
              <img
                src={image.img_url}
                alt={image.title}
                className="h-64 w-full object-cover transition-transform duration-300 group-hover:scale-105"
              />
              <div className="absolute inset-0 flex items-end bg-gradient-to-t from-black/50 to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100">
                <p className="px-3 pb-3 text-[13px] font-medium text-white line-clamp-2">
                  {image.title}
                </p>
              </div>
            </button>
          ))}
        </div>
      )}

      {/* ── Pagination ──────────────────────────────────────── */}
      {!loading && totalPages > 1 && (
        <div className="flex items-center justify-center gap-3 py-6">
          <button
            onClick={() => setPage((p) => p - 1)}
            disabled={page === 1}
            className="rounded-lg border border-[#CFEFD9] bg-white px-4 py-2 text-sm font-medium text-[#1F8F50] hover:bg-[#F0FFF6] disabled:cursor-not-allowed disabled:opacity-40"
          >
            Previous
          </button>
          <span className="text-sm text-[#687280]">
            Page {page} of {totalPages} &nbsp;·&nbsp; {total} photos
          </span>
          <button
            onClick={() => setPage((p) => p + 1)}
            disabled={page === totalPages}
            className="rounded-lg border border-[#CFEFD9] bg-white px-4 py-2 text-sm font-medium text-[#1F8F50] hover:bg-[#F0FFF6] disabled:cursor-not-allowed disabled:opacity-40"
          >
            Next
          </button>
        </div>
      )}

      {/* ── Preview modal ────────────────────────────────────── */}
      {selectedImage && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="relative mx-4 w-full max-w-2xl rounded-2xl bg-white p-6 shadow-2xl">
            <button
              type="button"
              onClick={() => setSelectedImage(null)}
              className="absolute right-4 top-4 flex h-8 w-8 items-center justify-center rounded-full bg-[#1F8F50] text-white transition hover:bg-[#2DBE6C]"
            >
              <X size={18} strokeWidth={3} />
            </button>

            <img
              src={selectedImage.img_url}
              alt={selectedImage.title}
              className="h-96 w-full rounded-xl object-cover"
            />

            <p className="mt-4 text-[16px] font-semibold text-gray-900">{selectedImage.title}</p>
            <p className="mt-1 text-[14px] text-gray-500">
              Uploaded on{" "}
              {new Date(selectedImage.created_at).toLocaleDateString("en-AU", {
                day: "2-digit",
                month: "2-digit",
                year: "numeric",
              })}
            </p>

            <div className="mt-4 flex justify-end gap-3">
              <button
                type="button"
                onClick={() => openEdit(selectedImage)}
                className="inline-flex items-center gap-2 rounded-lg bg-[#1F8F50] px-5 py-2 text-[14px] font-medium text-white transition hover:bg-[#2DBE6C]"
              >
                <Pencil size={14} />
                Edit
              </button>
              <button
                type="button"
                onClick={() => setDeleteTarget(selectedImage)}
                className="rounded-lg bg-red-500 px-5 py-2 text-[14px] font-medium text-white transition hover:bg-red-600"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Upload modal ─────────────────────────────────────── */}
      {uploadOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="relative mx-4 w-full max-w-md rounded-2xl bg-white p-6 shadow-2xl">
            <button
              type="button"
              onClick={closeUpload}
              className="absolute right-4 top-4 text-gray-400 transition hover:text-gray-700"
            >
              <X size={20} />
            </button>

            <h2 className="mb-5 text-[20px] font-semibold text-gray-900">Upload New Photo</h2>

            <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-[#1F8F50]/40 bg-[#E8F5EE] py-10 transition hover:border-[#1F8F50] hover:bg-[#D6EFE2]">
              {uploadPreview ? (
                <img src={uploadPreview} alt="Preview" className="max-h-40 rounded-lg object-contain" />
              ) : (
                <>
                  <Upload size={36} className="mb-3 text-[#1F8F50]" />
                  <p className="text-[17px] font-bold text-[#1F8F50]">Click to select a photo</p>
                  <p className="mt-1 text-[14px] font-bold text-[#1F8F50]/70">PNG, JPG, WEBP — max 5 MB</p>
                </>
              )}
              <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
            </label>

            <input
              type="text"
              placeholder="Photo title"
              value={uploadTitle}
              onChange={(e) => setUploadTitle(e.target.value)}
              maxLength={200}
              className="mt-4 w-full rounded-xl border-2 border-gray-300 bg-gray-100 px-4 py-3 text-[17px] font-bold text-gray-800 outline-none transition placeholder:font-bold placeholder:text-gray-400 focus:border-[#1F8F50] focus:bg-white focus:ring-2 focus:ring-[#1F8F50]/20"
            />

            <div className="mt-5 flex justify-end gap-3">
              <button
                type="button"
                onClick={closeUpload}
                className="rounded-lg border border-gray-200 px-5 py-2.5 text-[14px] font-medium text-gray-600 transition hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleUpload}
                disabled={!uploadTitle.trim() || !uploadFile || uploading}
                className="rounded-lg bg-[#1F8F50] px-5 py-2.5 text-[14px] font-medium text-white transition hover:bg-[#2DBE6C] disabled:opacity-40"
              >
                {uploading ? "Uploading…" : "Upload"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Edit modal ───────────────────────────────────────── */}
      {editTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="relative mx-4 w-full max-w-md rounded-2xl bg-white p-6 shadow-2xl">
            <button
              type="button"
              onClick={closeEdit}
              className="absolute right-4 top-4 text-gray-400 transition hover:text-gray-700"
            >
              <X size={20} />
            </button>

            <h2 className="mb-5 text-[20px] font-semibold text-gray-900">Edit Photo</h2>

            {/* Image replacement zone */}
            <div className="relative">
              <label
                className={`flex cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl border-2 border-dashed transition ${
                  editPreview
                    ? "border-[#1F8F50] bg-[#D6EFE2]"
                    : "border-[#1F8F50]/40 bg-[#E8F5EE] hover:border-[#1F8F50] hover:bg-[#D6EFE2]"
                }`}
              >
                <img
                  src={editPreview ?? editTarget.img_url}
                  alt="Preview"
                  className="max-h-48 w-full rounded-lg object-contain py-4 px-4"
                />
                <p className="pb-3 text-[12px] font-medium text-[#1F8F50]/70">
                  {editPreview ? "New image selected — click to change" : "Click to replace image (optional)"}
                </p>
                <input type="file" accept="image/*" className="hidden" onChange={handleEditFileChange} />
              </label>

              {/* Badge + clear button when a new file is staged */}
              {editPreview && (
                <div className="absolute left-2 top-2 flex items-center gap-1.5">
                  <span className="rounded-full bg-[#1F8F50] px-2.5 py-0.5 text-[11px] font-semibold text-white shadow">
                    New image
                  </span>
                  <button
                    type="button"
                    title="Revert to current image"
                    onClick={(e) => {
                      e.preventDefault();
                      setEditFile(null);
                      setEditPreview(null);
                    }}
                    className="flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-white shadow transition hover:bg-red-600"
                  >
                    <X size={10} strokeWidth={3} />
                  </button>
                </div>
              )}
            </div>

            {/* Title input + char counter */}
            <div className="mt-4">
              <input
                type="text"
                placeholder="Photo title"
                value={editTitle}
                onChange={(e) => setEditTitle(e.target.value)}
                maxLength={200}
                className="w-full rounded-xl border-2 border-gray-300 bg-gray-100 px-4 py-3 text-[15px] font-semibold text-gray-800 outline-none transition placeholder:font-normal placeholder:text-gray-400 focus:border-[#1F8F50] focus:bg-white focus:ring-2 focus:ring-[#1F8F50]/20"
              />
              <p className="mt-1 text-right text-[12px] text-gray-400">
                {editTitle.length}/200
              </p>
            </div>

            <div className="mt-4 flex justify-end gap-3">
              <button
                type="button"
                onClick={closeEdit}
                className="rounded-lg border border-gray-200 px-5 py-2.5 text-[14px] font-medium text-gray-600 transition hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSaveEdit}
                disabled={!hasEditChanges || saving}
                className="rounded-lg bg-[#1F8F50] px-5 py-2.5 text-[14px] font-medium text-white transition hover:bg-[#2DBE6C] disabled:opacity-40"
              >
                {saving ? "Saving…" : "Save Changes"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Delete confirmation ──────────────────────────────── */}
      <ConfirmModal
        open={!!deleteTarget}
        title="Delete Photo"
        message="Are you sure you want to delete this photo? This action cannot be undone."
        confirmText="Delete"
        cancelText="Cancel"
        isDanger
        onConfirm={confirmDelete}
        onCancel={() => setDeleteTarget(null)}
      />

      {toast && (
        <AdminToast message={toast.message} type={toast.type} onClose={() => setToast(null)} />
      )}
    </div>
  );
}
