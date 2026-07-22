"use client";

import { useEffect, useRef, useState } from "react";
import { ImagePlus, Save, X } from "lucide-react";

export default function BlogForm({ initialData, onSubmit, submitting = false }: any) {
  const [form, setForm] = useState(
    initialData || {
      title: "",
      description: "",
      date: "",
      content: "",
      coverImage: "",
    }
  );

  // initialData.coverImage may be a URL string (edit mode) or empty
  const [coverPreview, setCoverPreview] = useState<string>(
    typeof initialData?.coverImage === "string" ? initialData.coverImage : ""
  );

  const [errors, setErrors] = useState<any>({});

  const [editorLoaded, setEditorLoaded] = useState(false);
  const [CKEditorComponent, setCKEditorComponent] = useState<any>(null);
  const [ClassicEditor, setClassicEditor] = useState<any>(null);

  const coverImageRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const loadEditor = async () => {
      const ckeditor = await import("@ckeditor/ckeditor5-react");
      const classicEditor = await import("@ckeditor/ckeditor5-build-classic");

      setCKEditorComponent(() => ckeditor.CKEditor);
      setClassicEditor(() => classicEditor.default);
      setEditorLoaded(true);
    };

    loadEditor();
  }, []);

  const handleCoverImage = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setErrors((prev: any) => ({ ...prev, coverImage: "Only image files are allowed" }));
      return;
    }

    const url = URL.createObjectURL(file);
    setCoverPreview(url);
    setForm((prev: any) => ({ ...prev, coverImage: file }));
    setErrors((prev: any) => ({ ...prev, coverImage: "" }));
  };

  const removeCoverImage = (e: React.MouseEvent) => {
    e.stopPropagation();
    setCoverPreview("");
    setForm((prev: any) => ({ ...prev, coverImage: "" }));
    if (coverImageRef.current) coverImageRef.current.value = "";
  };

  const removeHtmlTags = (html: string) =>
    html.replace(/<[^>]*>/g, "").replace(/&nbsp;/g, " ").trim();

  const validateForm = () => {
    const newErrors: any = {};

    if (!form.title.trim()) newErrors.title = "Title is required";
    if (!form.description.trim()) newErrors.description = "Description is required";
    if (!form.date) newErrors.date = "Date is required";
    if (!removeHtmlTags(form.content)) newErrors.content = "Content is required";
    if (!form.coverImage) newErrors.coverImage = "Cover image is required";

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: any) => {
    e.preventDefault();
    if (!validateForm()) return;
    onSubmit(form);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Title */}
      <div>
        <label className="mb-2 block text-sm font-medium">Title</label>
        <input
          type="text"
          value={form.title}
          onChange={(e) => setForm({ ...form, title: e.target.value })}
          placeholder="Enter blog title"
          className={`w-full rounded-xl border bg-[#F9FAFB] px-4 py-3 outline-none focus:border-[#2DBE6C] focus:bg-white ${
            errors.title ? "border-red-500" : "border-[#E5E7EB]"
          }`}
        />
        {errors.title && <p className="mt-1 text-sm text-red-500">{errors.title}</p>}
      </div>

      {/* Description */}
      <div>
        <label className="mb-2 block text-sm font-medium">Description</label>
        <input
          type="text"
          value={form.description}
          onChange={(e) => setForm({ ...form, description: e.target.value })}
          placeholder="Short description or subtitle"
          className={`w-full rounded-xl border bg-[#F9FAFB] px-4 py-3 outline-none focus:border-[#2DBE6C] focus:bg-white ${
            errors.description ? "border-red-500" : "border-[#E5E7EB]"
          }`}
        />
        {errors.description && <p className="mt-1 text-sm text-red-500">{errors.description}</p>}
      </div>

      {/* Date */}
      <div>
        <label className="mb-2 block text-sm font-medium">Date</label>
        <input
          type="date"
          value={form.date}
          onChange={(e) => setForm({ ...form, date: e.target.value })}
          className={`w-full rounded-xl border bg-[#F9FAFB] px-4 py-3 outline-none focus:border-[#2DBE6C] focus:bg-white ${
            errors.date ? "border-red-500" : "border-[#E5E7EB]"
          }`}
        />
        {errors.date && <p className="mt-1 text-sm text-red-500">{errors.date}</p>}
      </div>

      {/* Cover Image */}
      <div>
        <label className="mb-3 block text-sm font-medium">Cover Image</label>

        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            if (e.dataTransfer.files?.[0]) handleCoverImage(e.dataTransfer.files[0]);
          }}
          onClick={() => coverImageRef.current?.click()}
          className={`relative cursor-pointer rounded-2xl border-2 border-dashed bg-[#F8FFFA] p-8 text-center hover:bg-[#F0FFF6] ${
            errors.coverImage ? "border-red-500" : "border-[#CFEFD9]"
          }`}
        >
          {coverPreview ? (
            <>
              <img
                src={coverPreview}
                alt="Cover preview"
                className="mx-auto h-48 rounded-lg object-cover"
              />
              <button
                type="button"
                onClick={removeCoverImage}
                className="absolute right-3 top-3 rounded-full bg-white p-1 shadow hover:bg-red-50"
              >
                <X size={16} className="text-red-500" />
              </button>
            </>
          ) : (
            <>
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-[#EAFBF0] text-[#1F8F50]">
                <ImagePlus size={22} />
              </div>
              <p className="mt-3 text-sm font-semibold">Upload cover image</p>
              <p className="text-sm text-[#687280]">Drag & drop or click to browse</p>
            </>
          )}
        </div>

        <input
          type="file"
          ref={coverImageRef}
          className="hidden"
          accept="image/*"
          onChange={(e) => e.target.files?.[0] && handleCoverImage(e.target.files[0])}
        />

        {errors.coverImage && <p className="mt-1 text-sm text-red-500">{errors.coverImage}</p>}
      </div>

      {/* Content — CKEditor */}
      <div>
        <label className="mb-2 block text-sm font-medium">Content</label>

        <div
          className={`overflow-hidden rounded-xl border bg-white ${
            errors.content ? "border-red-500" : "border-[#E5E7EB]"
          }`}
        >
          {editorLoaded && CKEditorComponent && ClassicEditor ? (
            <CKEditorComponent
              editor={ClassicEditor}
              data={form.content}
              onChange={(_event: any, editor: any) => {
                const data = editor.getData();
                setForm((prev: any) => ({ ...prev, content: data }));
                setErrors((prev: any) => ({ ...prev, content: "" }));
              }}
            />
          ) : (
            <div className="rounded-xl bg-[#F9FAFB] px-4 py-3 text-sm text-[#687280]">
              Loading editor...
            </div>
          )}
        </div>

        {errors.content && <p className="mt-1 text-sm text-red-500">{errors.content}</p>}
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={submitting}
        className="flex items-center gap-2 rounded-full bg-[#2DBE6C] px-6 py-3 text-white hover:bg-[#1F8F50] disabled:cursor-not-allowed disabled:opacity-60"
      >
        <Save size={18} />
        {submitting ? "Saving..." : "Save Blog"}
      </button>
    </form>
  );
}
