"use client";

import { useRef, useState } from "react";
import { ImagePlus, Save } from "lucide-react";

export default function BlogForm({ initialData, onSubmit }: any) {
  const [form, setForm] = useState(
    initialData || {
      title: "",
      subTitle: "",
      date: "",
      description: "",
      image: "",
    }
  );

  const [preview, setPreview] = useState(form.image || "");
  const [errors, setErrors] = useState<any>({});
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImage = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setErrors((prev: any) => ({
        ...prev,
        image: "Only image files are allowed",
      }));
      return;
    }

    const imageUrl = URL.createObjectURL(file);
    setPreview(imageUrl);

    setForm({
      ...form,
      image: file,
    });

    setErrors((prev: any) => ({
      ...prev,
      image: "",
    }));
  };

  const validateForm = () => {
    const newErrors: any = {};

    if (!form.title.trim()) {
      newErrors.title = "Title is required";
    }

    if (!form.subTitle.trim()) {
      newErrors.subTitle = "Sub title is required";
    }

    if (!form.date) {
      newErrors.date = "Date is required";
    }

    if (!form.description.trim()) {
      newErrors.description = "Description is required";
    }

    if (!form.image) {
      newErrors.image = "Image is required";
    }

    setErrors(newErrors);

    return Object.keys(newErrors).length === 0;
  };

  const handleDrop = (e: any) => {
    e.preventDefault();

    if (e.dataTransfer.files?.[0]) {
      handleImage(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = (e: any) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    onSubmit(form);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Enter Title */}
      <div>
        <label className="mb-2 block text-sm font-medium">Enter Title</label>
        <input
          type="text"
          value={form.title}
          onChange={(e) => setForm({ ...form, title: e.target.value })}
          placeholder="Enter title"
          className={`w-full rounded-xl border bg-[#F9FAFB] px-4 py-3 outline-none focus:border-[#2DBE6C] focus:bg-white ${
            errors.title ? "border-red-500" : "border-[#E5E7EB]"
          }`}
        />
        {errors.title && (
          <p className="mt-1 text-sm text-red-500">{errors.title}</p>
        )}
      </div>

      {/* Enter Sub Title */}
      <div>
        <label className="mb-2 block text-sm font-medium">
          Enter Sub Title
        </label>
        <input
          type="text"
          value={form.subTitle}
          onChange={(e) => setForm({ ...form, subTitle: e.target.value })}
          placeholder="Enter sub title"
          className={`w-full rounded-xl border bg-[#F9FAFB] px-4 py-3 outline-none focus:border-[#2DBE6C] focus:bg-white ${
            errors.subTitle ? "border-red-500" : "border-[#E5E7EB]"
          }`}
        />
        {errors.subTitle && (
          <p className="mt-1 text-sm text-red-500">{errors.subTitle}</p>
        )}
      </div>

      {/* Select Date */}
      <div>
        <label className="mb-2 block text-sm font-medium">Select Date</label>
        <input
          type="date"
          value={form.date}
          onChange={(e) => setForm({ ...form, date: e.target.value })}
          className={`w-full rounded-xl border bg-[#F9FAFB] px-4 py-3 outline-none focus:border-[#2DBE6C] focus:bg-white ${
            errors.date ? "border-red-500" : "border-[#E5E7EB]"
          }`}
        />
        {errors.date && (
          <p className="mt-1 text-sm text-red-500">{errors.date}</p>
        )}
      </div>

      {/* Description */}
      <div>
        <label className="mb-2 block text-sm font-medium">Description</label>
        <textarea
          rows={5}
          value={form.description}
          onChange={(e) => setForm({ ...form, description: e.target.value })}
          placeholder="Enter description"
          className={`w-full rounded-xl border bg-[#F9FAFB] px-4 py-3 outline-none focus:border-[#2DBE6C] focus:bg-white ${
            errors.description ? "border-red-500" : "border-[#E5E7EB]"
          }`}
        />
        {errors.description && (
          <p className="mt-1 text-sm text-red-500">{errors.description}</p>
        )}
      </div>

      {/* Image Upload */}
      <div>
        <label className="mb-3 block text-sm font-medium">Upload Image</label>

        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`cursor-pointer rounded-2xl border-2 border-dashed bg-[#F8FFFA] p-8 text-center hover:bg-[#F0FFF6] ${
            errors.image ? "border-red-500" : "border-[#CFEFD9]"
          }`}
        >
          {!preview ? (
            <>
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-[#EAFBF0] text-[#1F8F50]">
                <ImagePlus size={22} />
              </div>
              <p className="mt-3 text-sm font-semibold">Upload image</p>
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
          onChange={(e) => e.target.files && handleImage(e.target.files[0])}
        />

        {errors.image && (
          <p className="mt-1 text-sm text-red-500">{errors.image}</p>
        )}
      </div>

      {/* Submit */}
      <button
        type="submit"
        className="flex items-center gap-2 rounded-full bg-[#2DBE6C] px-6 py-3 text-white hover:bg-[#1F8F50]"
      >
        <Save size={18} />
        Save Blog
      </button>
    </form>
  );
}