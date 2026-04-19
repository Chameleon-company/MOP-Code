"use client";

import { useState, useRef } from "react";
import { Save, ImagePlus } from "lucide-react";

export default function UseCaseForm({ initialData, onSubmit }: any) {
  const [form, setForm] = useState(
    initialData || {
      serialNumber: "",
      title: "",
      category: "",
      description: "",
      image: "",
      document: null,
    }
  );

  const [preview, setPreview] = useState(form.image || "");

  const fileInputRef = useRef<HTMLInputElement>(null);
  const documentInputRef = useRef<HTMLInputElement>(null); 

  // image handler
  const handleFile = (file: File) => {
    const imageUrl = URL.createObjectURL(file);
    setPreview(imageUrl);

    setForm({
      ...form,
      image: file,
    });
  };

  // document handler
  const handleDocument = (file: File) => {
    const allowedTypes = ["text/html", "application/pdf"];

    if (!allowedTypes.includes(file.type)) {
      alert("Only HTML and PDF files are allowed");
      return;
    }

    setForm({
      ...form,
      document: file,
    });
  };

  // drop handler
  const handleDrop = (e: any) => {
    e.preventDefault();
    if (e.dataTransfer.files?.[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  // submit
  const handleSubmit = (e: any) => {
    e.preventDefault();
    onSubmit(form);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">

      {/* Serial Number */}
      <div>
        <label className="mb-2 block text-sm font-medium">
          Serial Number
        </label>
        <input
          type="text"
          value={form.serialNumber}
          onChange={(e) =>
            setForm({ ...form, serialNumber: e.target.value })
          }
          placeholder="Enter serial number"
          className="w-full rounded-xl border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 focus:border-[#2DBE6C] focus:bg-white outline-none"
        />
      </div>

      {/* Title */}
      <div>
        <label className="mb-2 block text-sm font-medium">
          Title
        </label>
        <input
          type="text"
          value={form.title}
          onChange={(e) =>
            setForm({ ...form, title: e.target.value })
          }
          placeholder="Enter title"
          className="w-full rounded-xl border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 focus:border-[#2DBE6C] focus:bg-white outline-none"
        />
      </div>

      {/* Category */}
      <div>
        <label className="mb-2 block text-sm font-medium">
          Category
        </label>
        <select
          value={form.category}
          onChange={(e) =>
            setForm({ ...form, category: e.target.value })
          }
          className="w-full rounded-xl border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 outline-none focus:border-[#2DBE6C]"
        >
          <option value="">Select category</option>
          <option value="Category 1">Category 1</option>
          <option value="Category 2">Category 2</option>
          <option value="Category 3">Category 3</option>
        </select>
      </div>

      {/* Description */}
      <div>
        <label className="mb-2 block text-sm font-medium">
          Description
        </label>
        <textarea
          rows={4}
          value={form.description}
          onChange={(e) =>
            setForm({ ...form, description: e.target.value })
          }
          placeholder="Enter description"
          className="w-full rounded-xl border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 focus:border-[#2DBE6C] focus:bg-white outline-none"
        />
      </div>

      {/* image upload*/}
      <div>
        <label className="mb-3 block text-sm font-medium">
          Cover Image
        </label>

        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className="cursor-pointer rounded-2xl border-2 border-dashed border-[#CFEFD9] bg-[#F8FFFA] p-8 text-center hover:bg-[#F0FFF6]"
        >
          {!preview ? (
            <>
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-[#EAFBF0] text-[#1F8F50]">
                <ImagePlus size={22} />
              </div>
              <p className="mt-3 text-sm font-semibold">
                Upload image
              </p>
              <p className="text-sm text-[#687280]">
                Drag & drop or click to browse
              </p>
            </>
          ) : (
            <img
              src={preview}
              alt="preview"
              className="mx-auto h-40 rounded-lg object-cover"
            />
          )}
        </div>

        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          accept="image/*"
          onChange={(e) =>
            e.target.files && handleFile(e.target.files[0])
          }
        />
      </div>

      {/* document upload */}
      <div>
        <label className="mb-3 block text-sm font-medium">
          Upload HTML / PDF
        </label>

        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            if (e.dataTransfer.files?.[0]) {
              handleDocument(e.dataTransfer.files[0]);
            }
          }}
          onClick={() => documentInputRef.current?.click()}
          className="cursor-pointer rounded-2xl border-2 border-dashed border-[#CFEFD9] bg-[#F8FFFA] p-8 text-center hover:bg-[#F0FFF6]"
        >
          {!form.document ? (
            <>
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-[#EAFBF0] text-[#1F8F50]">
                📄
              </div>
              <p className="mt-3 text-sm font-semibold">
                Upload HTML or PDF
              </p>
              <p className="text-sm text-[#687280]">
                Drag & drop or click to browse
              </p>
            </>
          ) : (
            <p className="text-sm font-medium text-[#1F8F50]">
              {form.document.name}
            </p>
          )}
        </div>

        <input
          type="file"
          ref={documentInputRef}
          className="hidden"
          accept=".html,.pdf"
          onChange={(e) =>
            e.target.files && handleDocument(e.target.files[0])
          }
        />
      </div>

      {/* Submit */}
      <button className="flex items-center gap-2 rounded-full bg-[#2DBE6C] px-6 py-3 text-white hover:bg-[#1F8F50]">
        <Save size={18} />
        Save Use Case
      </button>

    </form>
  );
}