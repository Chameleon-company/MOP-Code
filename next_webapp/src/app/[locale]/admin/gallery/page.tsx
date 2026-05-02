"use client";

import { useState } from "react";
import { Plus, X, Upload } from "lucide-react";

type GalleryImage = {
  id: number;
  title: string;
  url: string;
  uploadedAt: string;
};

const initialImages: GalleryImage[] = [
  { id: 1, title: "Flinders Street Station, Melbourne, Australia", url: "/img/melbourne-city.jpg", uploadedAt: "02/04/2026" },
  { id: 2, title: "Melbourne Skyline", url: "/img/melbourne-city1.jpg", uploadedAt: "01/04/2026" },
  { id: 3, title: "Royal Botanic Gardens", url: "/img/royalBotanic.jpg", uploadedAt: "31/03/2026" },
  { id: 4, title: "Southern Cross Station", url: "/img/southernCross.jpg", uploadedAt: "30/03/2026" },
  { id: 5, title: "Docklands, Melbourne", url: "/img/docklands.jpg", uploadedAt: "29/03/2026" },
  { id: 6, title: "City View", url: "/img/cityimg.png", uploadedAt: "28/03/2026" },
  { id: 7, title: "Melbourne Arts Centre", url: "/img/melbourne-city.jpg", uploadedAt: "27/03/2026" },
  { id: 8, title: "Federation Square", url: "/img/melbourne-city1.jpg", uploadedAt: "26/03/2026" },
];

export default function GalleryPage() {
  const [images, setImages] = useState<GalleryImage[]>(initialImages);
  const [selectedImage, setSelectedImage] = useState<GalleryImage | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadTitle, setUploadTitle] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadPreview, setUploadPreview] = useState<string | null>(null);

  const handleDelete = () => {
    if (!selectedImage) return;
    setImages((prev) => prev.filter((img) => img.id !== selectedImage.id));
    setShowDeleteConfirm(false);
    setSelectedImage(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadFile(file);
    setUploadPreview(URL.createObjectURL(file));
  };

  const handleUpload = () => {
    if (!uploadTitle.trim() || !uploadPreview) return;
    const newImage: GalleryImage = {
      id: Date.now(),
      title: uploadTitle.trim(),
      url: uploadPreview,
      uploadedAt: new Date().toLocaleDateString("en-AU"),
    };
    setImages((prev) => [newImage, ...prev]);
    setUploadOpen(false);
    setUploadTitle("");
    setUploadFile(null);
    setUploadPreview(null);
  };

  const closeUpload = () => {
    setUploadOpen(false);
    setUploadTitle("");
    setUploadFile(null);
    setUploadPreview(null);
  };

  return (
    /* Break out of the admin layout's p-8 to fill the full content area */
    <div className="-m-8">
      {/* ── Page header ─────────────────────────────────────── */}
      <div className="flex items-center justify-between px-8 py-6">
        <h1 className="text-[40px] font-semibold leading-[48px] text-[#2DBE6C]">
          Gallery
        </h1>
        <button
          type="button"
          onClick={() => setUploadOpen(true)}
          className="inline-flex items-center gap-2 rounded-lg bg-[#1F8F50] px-5 py-3 text-[14px] font-medium text-white transition hover:bg-[#2DBE6C]"
        >
          <Plus size={16} />
          Upload New Photo
        </button>
      </div>

      {/* ── Photo grid ──────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-1 sm:grid-cols-3 lg:grid-cols-4">
        {images.map((image) => (
          <button
            key={image.id}
            type="button"
            onClick={() => setSelectedImage(image)}
            className="group relative overflow-hidden focus:outline-none"
          >
            <img
              src={image.url}
              alt={image.title}
              className="h-64 w-full object-cover transition-transform duration-300 group-hover:scale-105"
            />
          </button>
        ))}

        {images.length === 0 && (
          <div className="col-span-4 flex flex-col items-center justify-center border-2 border-dashed border-gray-200 bg-white py-32 text-gray-400">
            <p className="text-[16px] font-medium">No photos yet</p>
            <p className="mt-1 text-[14px]">Click &ldquo;Upload New Photo&rdquo; to get started</p>
          </div>
        )}
      </div>

      {/* ── Preview modal ────────────────────────────────────── */}
      {selectedImage && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="relative w-full max-w-2xl rounded-2xl bg-white p-6 shadow-2xl mx-4">
            {/* Close */}
            <button
              type="button"
              onClick={() => { setSelectedImage(null); setShowDeleteConfirm(false); }}
              className="absolute right-4 top-4 flex h-8 w-8 items-center justify-center rounded-full bg-[#1F8F50] text-white transition hover:bg-[#2DBE6C]"
            >
              <X size={18} strokeWidth={3} />
            </button>

            {/* Image */}
            <img
              src={selectedImage.url}
              alt={selectedImage.title}
              className="h-96 w-full rounded-xl object-cover"
            />

            {/* Title + date */}
            <p className="mt-4 text-[16px] font-semibold text-gray-900">
              {selectedImage.title}
            </p>
            <p className="mt-1 text-[15px] font-semibold text-gray-500">
              Uploaded on {selectedImage.uploadedAt}
            </p>

            {/* Delete button */}
            <div className="mt-4 flex justify-end">
              <button
                type="button"
                onClick={() => setShowDeleteConfirm(true)}
                className="rounded-lg bg-red-500 px-5 py-2 text-[14px] font-medium text-white transition hover:bg-red-600"
              >
                Delete
              </button>
            </div>

            {/* ── Delete confirmation dialog ───────────────── */}
            {showDeleteConfirm && (
              <div className="absolute inset-0 flex items-center justify-center rounded-2xl bg-black/40">
                <div className="relative w-64 rounded-2xl bg-white px-6 py-5 shadow-xl">
                  <button
                    type="button"
                    onClick={() => setShowDeleteConfirm(false)}
                    className="absolute right-3 top-3 text-gray-400 transition hover:text-gray-700"
                  >
                    <X size={16} />
                  </button>
                  <p className="text-center text-[15px] font-semibold leading-snug text-gray-900">
                    Are you sure you want to delete this photo?
                  </p>
                  <div className="mt-4 flex items-center justify-center gap-3">
                    <button
                      type="button"
                      onClick={() => setShowDeleteConfirm(false)}
                      className="rounded-lg border border-gray-300 px-4 py-2 text-[13px] font-medium text-gray-600 transition hover:bg-gray-50"
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      onClick={handleDelete}
                      className="rounded-lg bg-red-500 px-4 py-2 text-[13px] font-semibold text-white transition hover:bg-red-600"
                    >
                      Yes, Delete
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Upload modal ─────────────────────────────────────── */}
      {uploadOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="relative w-full max-w-md rounded-2xl bg-white p-6 shadow-2xl mx-4">
            {/* Close */}
            <button
              type="button"
              onClick={closeUpload}
              className="absolute right-4 top-4 text-gray-400 transition hover:text-gray-700"
            >
              <X size={20} />
            </button>

            <h2 className="mb-5 text-[20px] font-semibold text-gray-900">
              Upload New Photo
            </h2>

            {/* File drop zone */}
            <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-[#1F8F50]/40 bg-[#E8F5EE] py-10 transition hover:border-[#1F8F50] hover:bg-[#D6EFE2]">
              {uploadPreview ? (
                <img
                  src={uploadPreview}
                  alt="Preview"
                  className="max-h-40 rounded-lg object-contain"
                />
              ) : (
                <>
                  <Upload size={36} className="mb-3 text-[#1F8F50]" />
                  <p className="text-[17px] font-bold text-[#1F8F50]">
                    Click to select a photo
                  </p>
                  <p className="mt-1 text-[14px] font-bold text-[#1F8F50]/70">PNG, JPG, WEBP — max 10MB</p>
                </>
              )}
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileChange}
              />
            </label>

            {/* Title input */}
            <input
              type="text"
              placeholder="Photo title"
              value={uploadTitle}
              onChange={(e) => setUploadTitle(e.target.value)}
              className="mt-4 w-full rounded-xl border-2 border-gray-300 bg-gray-100 px-4 py-3 text-[17px] font-bold text-gray-800 outline-none transition placeholder:text-gray-400 placeholder:font-bold focus:border-[#1F8F50] focus:bg-white focus:ring-2 focus:ring-[#1F8F50]/20"
            />

            {/* Actions */}
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
                disabled={!uploadTitle.trim() || !uploadPreview}
                className="rounded-lg bg-[#1F8F50] px-5 py-2.5 text-[14px] font-medium text-white transition hover:bg-[#2DBE6C] disabled:opacity-40"
              >
                Upload
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
