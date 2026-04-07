"use client";

import { FolderPlus, ImagePlus, Save } from "lucide-react";

export default function AddCategoryForm() {
  return (
    <div className="rounded-3xl border border-[#E5E7EB] bg-white p-8 shadow-sm">
      
      {/* Header */}
      <div className="mb-7 flex items-center gap-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-[#EAFBF0] text-[#1F8F50]">
          <FolderPlus size={22} />
        </div>

        <div>
          <h2 className="text-xl font-semibold text-[#1A1A1A]">
            Add Category
          </h2>
          <p className="text-sm text-[#687280]">
            Create a new category for your website content
          </p>
        </div>
      </div>

      <form className="space-y-6">

        {/* Title */}
        <div>
          <label className="mb-2 block text-sm font-medium text-[#1A1A1A]">
            Title
          </label>

          <input
            type="text"
            placeholder="Enter category title"
            className="w-full rounded-xl border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 text-sm outline-none transition focus:border-[#2DBE6C] focus:bg-white"
          />
        </div>

        {/* Description */}
        <div>
          <label className="mb-2 block text-sm font-medium text-[#1A1A1A]">
            Description
          </label>

          <textarea
            rows={4}
            placeholder="Enter category description"
            className="w-full resize-none rounded-xl border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 text-sm outline-none transition focus:border-[#2DBE6C] focus:bg-white"
          />
        </div>

        {/* Image Upload */}
        <div>
          <label className="mb-3 block text-sm font-medium text-[#1A1A1A]">
            Category Image
          </label>

          <div className="rounded-2xl border-2 border-dashed border-[#CFEFD9] bg-[#F8FFFA] p-8 text-center">

            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-[#EAFBF0] text-[#1F8F50]">
              <ImagePlus size={22} />
            </div>

            <h3 className="mt-3 text-sm font-semibold text-[#1A1A1A]">
              Upload category image
            </h3>

            <p className="mt-1 text-sm text-[#687280]">
              Drag and drop your image here or click to browse
            </p>

            <button
              type="button"
              className="mt-4 rounded-full border border-[#2DBE6C] px-4 py-2 text-sm font-medium text-[#1F8F50] transition hover:bg-[#EAFBF0]"
            >
              Choose File
            </button>
          </div>
        </div>

        {/* Button */}
        <div className="pt-2">
          <button
            type="submit"
            className="inline-flex items-center gap-2 rounded-full bg-[#2DBE6C] px-6 py-3 text-sm font-medium text-white transition hover:bg-[#1F8F50]"
          >
            <Save size={18} />
            Save Category
          </button>
        </div>

      </form>

    </div>
  );
}