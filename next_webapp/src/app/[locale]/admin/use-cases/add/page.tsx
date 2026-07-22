// "use client";

// import { useEffect, useRef, useState } from "react";
// import { useRouter, useParams } from "next/navigation";
// import { BookOpen, ImagePlus, Save, X } from "lucide-react";

// function getAuthHeaders() {
//   const user = JSON.parse(localStorage.getItem("user") || "{}");
//   const userId = user.userId ?? user.id ?? "";
//   const roleId = user.roleId ?? user.role_id ?? "";
//   const token = user.token ?? "";
//   return {
//     "x-user-id": String(userId),
//     "x-user-role-id": String(roleId),
//     "x-user-role": user.roleName ?? user.role_name ?? "",
//     ...(token ? { Authorization: `Bearer ${token}` } : {}),
//   };
// }

// export default function AddUseCasePage() {
//   const router = useRouter();
//   const { locale } = useParams() as { locale: string };

//   const [title, setTitle] = useState("");
//   const [description, setDescription] = useState("");
//   const [categoryId, setCategoryId] = useState("");
//   const [categories, setCategories] = useState<any[]>([]);
//   const [tags, setTags] = useState<string[]>([]);
//   const [tagInput, setTagInput] = useState("");
//   const [imageFile, setImageFile] = useState<File | null>(null);
//   const [imagePreview, setImagePreview] = useState<string | null>(null);
//   const [saving, setSaving] = useState(false);
//   const [error, setError] = useState("");

//   const fileInputRef = useRef<HTMLInputElement>(null);

//   useEffect(() => {
//     fetch("/api/categories", { headers: getAuthHeaders() })
//       .then((r) => r.json())
//       .then((json) => {
//         if (json.success) setCategories(json.data || []);
//       });
//   }, []);

//   function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
//     const file = e.target.files?.[0];
//     if (!file) return;
//     setImageFile(file);
//     setImagePreview(URL.createObjectURL(file));
//   }

//   function addTag() {
//     const t = tagInput.trim();
//     if (t && !tags.includes(t)) setTags((prev) => [...prev, t]);
//     setTagInput("");
//   }

//   function handleTagKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
//     if (e.key === "Enter" || e.key === ",") {
//       e.preventDefault();
//       addTag();
//     }
//   }

//   async function handleSubmit(e: React.FormEvent) {
//     e.preventDefault();
//     setSaving(true);
//     setError("");

//     try {
//       const authHeaders = getAuthHeaders();
//       const user = JSON.parse(localStorage.getItem("user") || "{}");
//       const userId = Number(user.userId ?? user.id ?? 0);

//       // Upload image if selected
//       let coverImgUrl: string | null = null;
//       if (imageFile) {
//         const formData = new FormData();
//         formData.append("file", imageFile);
//         formData.append("folder", "usecases");
//         formData.append("bucket", "usecase-images");

//         const uploadRes = await fetch("/api/upload", {
//           method: "POST",
//           headers: authHeaders,
//           body: formData,
//         });
//         const uploadJson = await uploadRes.json();

//         if (!uploadJson.success) {
//           setError("Image upload failed: " + (uploadJson.message || "Unknown error"));
//           setSaving(false);
//           return;
//         }
//         coverImgUrl = uploadJson.url;
//       }

//       const res = await fetch("/api/usecases", {
//         method: "POST",
//         headers: { "Content-Type": "application/json", ...authHeaders },
//         body: JSON.stringify({
//           title,
//           description,
//           cover_img: coverImgUrl,
//           category_id: categoryId ? Number(categoryId) : null,
//           created_by: userId,
//           tags,
//         }),
//       });
//       const json = await res.json();

//       if (!json.success) {
//         setError(json.message || json.error || "Failed to create use case.");
//         setSaving(false);
//         return;
//       }

//       router.push(`/${locale}/admin/use-cases`);
//     } catch {
//       setError("Failed to create use case.");
//       setSaving(false);
//     }
//   }

//   return (
//     <div className="rounded-3xl border border-[#E5E7EB] bg-white p-8 shadow-sm">
//       {/* Header */}
//       <div className="mb-7 flex items-center gap-3">
//         <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-[#EAFBF0] text-[#1F8F50]">
//           <BookOpen size={22} />
//         </div>
//         <div>
//           <h2 className="text-xl font-semibold text-[#1A1A1A]">Add Use Case</h2>
//           <p className="text-sm text-[#687280]">Create a new use case</p>
//         </div>
//       </div>

//       {error && (
//         <div className="mb-5 rounded-xl bg-red-50 px-4 py-3 text-sm text-red-600">
//           {error}
//         </div>
//       )}

//       <form onSubmit={handleSubmit} className="space-y-6">
//         {/* Title */}
//         <div>
//           <label className="mb-2 block text-sm font-medium text-[#1A1A1A]">
//             Title <span className="text-red-500">*</span>
//           </label>
//           <input
//             type="text"
//             required
//             value={title}
//             onChange={(e) => setTitle(e.target.value)}
//             placeholder="Enter use case title"
//             className="w-full rounded-xl border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 text-sm outline-none transition focus:border-[#2DBE6C] focus:bg-white"
//           />
//         </div>

//         {/* Category */}
//         <div>
//           <label className="mb-2 block text-sm font-medium text-[#1A1A1A]">
//             Category
//           </label>
//           <select
//             value={categoryId}
//             onChange={(e) => setCategoryId(e.target.value)}
//             className="w-full rounded-xl border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 text-sm outline-none focus:border-[#2DBE6C] focus:bg-white"
//           >
//             <option value="">Select a category</option>
//             {categories.map((c) => (
//               <option key={c.id} value={c.id}>
//                 {c.category_name}
//               </option>
//             ))}
//           </select>
//         </div>

//         {/* Description */}
//         <div>
//           <label className="mb-2 block text-sm font-medium text-[#1A1A1A]">
//             Description
//           </label>
//           <textarea
//             rows={4}
//             value={description}
//             onChange={(e) => setDescription(e.target.value)}
//             placeholder="Enter description"
//             className="w-full resize-none rounded-xl border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 text-sm outline-none transition focus:border-[#2DBE6C] focus:bg-white"
//           />
//         </div>

//         {/* Tags */}
//         <div>
//           <label className="mb-2 block text-sm font-medium text-[#1A1A1A]">
//             Tags
//           </label>
//           {tags.length > 0 && (
//             <div className="mb-3 flex flex-wrap gap-2">
//               {tags.map((tag) => (
//                 <span
//                   key={tag}
//                   className="inline-flex items-center gap-1 rounded-full bg-[#EAFBF0] px-3 py-1 text-xs font-medium text-[#1F8F50]"
//                 >
//                   {tag}
//                   <button
//                     type="button"
//                     onClick={() => setTags((prev) => prev.filter((t) => t !== tag))}
//                     className="ml-1 hover:text-[#1A6633]"
//                   >
//                     <X size={11} />
//                   </button>
//                 </span>
//               ))}
//             </div>
//           )}
//           <div className="flex gap-2">
//             <input
//               type="text"
//               value={tagInput}
//               onChange={(e) => setTagInput(e.target.value)}
//               onKeyDown={handleTagKeyDown}
//               placeholder="Type a tag and press Enter or comma"
//               className="flex-1 rounded-xl border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 text-sm outline-none transition focus:border-[#2DBE6C] focus:bg-white"
//             />
//             <button
//               type="button"
//               onClick={addTag}
//               className="rounded-xl border border-[#2DBE6C] px-4 py-3 text-sm font-medium text-[#2DBE6C] transition hover:bg-[#EAFBF0]"
//             >
//               Add
//             </button>
//           </div>
//         </div>

//         {/* Cover Image */}
//         <div>
//           <label className="mb-3 block text-sm font-medium text-[#1A1A1A]">
//             Cover Image
//           </label>
//           <div
//             onClick={() => fileInputRef.current?.click()}
//             className="cursor-pointer rounded-2xl border-2 border-dashed border-[#CFEFD9] bg-[#F8FFFA] p-8 text-center transition hover:bg-[#F0FFF6]"
//           >
//             {imagePreview ? (
//               <img
//                 src={imagePreview}
//                 alt="Preview"
//                 className="mx-auto h-40 rounded-lg object-cover"
//               />
//             ) : (
//               <>
//                 <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-[#EAFBF0] text-[#1F8F50]">
//                   <ImagePlus size={22} />
//                 </div>
//                 <h3 className="mt-3 text-sm font-semibold text-[#1A1A1A]">
//                   Upload cover image
//                 </h3>
//                 <p className="mt-1 text-sm text-[#687280]">
//                   Click to browse · JPEG, PNG, WebP · max 5 MB
//                 </p>
//               </>
//             )}
//           </div>
//           <input
//             ref={fileInputRef}
//             type="file"
//             accept="image/*"
//             className="hidden"
//             onChange={handleFileChange}
//           />
//           {imagePreview && (
//             <button
//               type="button"
//               onClick={() => {
//                 setImageFile(null);
//                 setImagePreview(null);
//               }}
//               className="mt-2 text-xs text-red-500 hover:underline"
//             >
//               Remove image
//             </button>
//           )}
//         </div>

//         {/* Actions */}
//         <div className="flex items-center gap-3 pt-2">
//           <button
//             type="submit"
//             disabled={saving}
//             className="inline-flex items-center gap-2 rounded-full bg-[#2DBE6C] px-6 py-3 text-sm font-medium text-white transition hover:bg-[#1F8F50] disabled:opacity-60"
//           >
//             <Save size={18} />
//             {saving ? "Saving..." : "Save Use Case"}
//           </button>
//           <button
//             type="button"
//             onClick={() => router.push(`/${locale}/admin/use-cases`)}
//             className="rounded-full border border-[#E5E7EB] px-6 py-3 text-sm font-medium text-[#687280] transition hover:bg-[#F9FAFB]"
//           >
//             Cancel
//           </button>
//         </div>
//       </form>
//     </div>
//   );
// }
"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import { BookOpen, ImagePlus, Save, X } from "lucide-react";
import AdminToast from "@/components/admin/AdminToast";
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

export default function AddUseCasePage() {
  const router = useRouter();
  const { locale } = useParams() as { locale: string };

  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [categoryId, setCategoryId] = useState("");
  const [categories, setCategories] = useState<any[]>([]);
  const [tags, setTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageUploading, setImageUploading] = useState(false);
  const imageUploadPromiseRef = useRef<Promise<string | null> | null>(null);
  const [notebookFile, setNotebookFile] = useState<File | null>(null);
  const [notebookContent, setNotebookContent] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [toast, setToast] = useState<{ message: string; type: "success" | "error" } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const notebookInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch("/api/categories", { headers: getAuthHeaders() })
      .then((r) => r.json())
      .then((json) => {
        if (json.success) setCategories(json.data || []);
      });
  }, []);

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

  function handleNotebookFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.name.endsWith(".ipynb")) {
      setError("Please upload a valid .ipynb file.");
      return;
    }
    setNotebookFile(file);
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      try {
        const parsed = JSON.parse(text);
        if (!Array.isArray(parsed.cells)) throw new Error();
        setNotebookContent(text);
        setError("");
      } catch {
        setError("Invalid notebook file — could not parse .ipynb JSON.");
        setNotebookFile(null);
        setNotebookContent(null);
      }
    };
    reader.readAsText(file);
  }

  function addTag() {
    const t = tagInput.trim();
    if (t && !tags.includes(t)) setTags((prev) => [...prev, t]);
    setTagInput("");
  }

  function handleTagKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      addTag();
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError("");

    try {
      const authHeaders = getAuthHeaders();
      const user = JSON.parse(localStorage.getItem("user") || "{}");
      const userId = Number(user.userId ?? user.id ?? 0);

      // Await the pre-upload that started when the user picked the file
      let coverImgUrl: string | null = null;
      if (imageFile && imageUploadPromiseRef.current) {
        coverImgUrl = await imageUploadPromiseRef.current;
        if (!coverImgUrl) {
          setError("Image upload failed. Please remove the image and try again.");
          setSaving(false);
          return;
        }
      }

      const res = await fetch("/api/usecases", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders },
        body: JSON.stringify({
          title,
          description,
          cover_img: coverImgUrl,
          content: notebookContent ?? null,
          category_id: categoryId ? Number(categoryId) : null,
          created_by: userId,
          tags,
        }),
      });
      const json = await res.json();

      if (!json.success) {
        const msg = json.message || json.error || "Failed to create use case.";
        setError(msg);
        setToast({ message: msg, type: "error" });
        setSaving(false);
        return;
      }
      
      setToast({
        message: "Use case added successfully.",
        type: "success",
      });
      
      setTimeout(() => {
        router.push(`/${locale}/admin/use-cases`);
      }, 1000);
    } catch {
      setError("Failed to create use case.");
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
          <h2 className="text-xl font-semibold text-[#1A1A1A]">Add Use Case</h2>
          <p className="text-sm text-[#687280]">Create a new use case</p>
        </div>
      </div>

      {error && (
        <div className="mb-5 rounded-xl bg-red-50 px-4 py-3 text-sm text-red-600">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
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
          <label className="mb-2 block text-sm font-medium text-[#1A1A1A]">
            Tags
          </label>
          {tags.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-2">
              {tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex items-center gap-1 rounded-full bg-[#EAFBF0] px-3 py-1 text-xs font-medium text-[#1F8F50]"
                >
                  {tag}
                  <button
                    type="button"
                    onClick={() => setTags((prev) => prev.filter((t) => t !== tag))}
                    className="ml-1 hover:text-[#1A6633]"
                  >
                    <X size={11} />
                  </button>
                </span>
              ))}
            </div>
          )}
          <div className="flex gap-2">
            <input
              type="text"
              value={tagInput}
              onChange={(e) => setTagInput(e.target.value)}
              onKeyDown={handleTagKeyDown}
              placeholder="Type a tag and press Enter or comma"
              className="flex-1 rounded-xl border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 text-sm outline-none transition focus:border-[#2DBE6C] focus:bg-white"
            />
            <button
              type="button"
              onClick={addTag}
              className="rounded-xl border border-[#2DBE6C] px-4 py-3 text-sm font-medium text-[#2DBE6C] transition hover:bg-[#EAFBF0]"
            >
              Add
            </button>
          </div>
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
                setImageUploading(false);
                imageUploadPromiseRef.current = null;
              }}
              className="mt-2 text-xs text-red-500 hover:underline"
            >
              Remove image
            </button>
          )}
        </div>

        {/* Python Notebook */}
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
              {notebookFile ? notebookFile.name : "Upload Python notebook"}
            </h3>
            <p className="mt-1 text-sm text-[#687280]">
              Click to browse · IPYNB file
            </p>
          </div>
          <input
            ref={notebookInputRef}
            type="file"
            accept=".ipynb,application/x-ipynb+json,application/json"
            className="hidden"
            onChange={handleNotebookFileChange}
          />
          {notebookFile && (
            <button
              type="button"
              onClick={() => { setNotebookFile(null); setNotebookContent(null); }}
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
              : "Save Use Case"}
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