"use client";

import { useState, type FormEvent } from "react";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft, ImagePlus, Save } from "lucide-react";

type UseCaseItem = {
  id: string;
  title: string;
  category: string;
  description: string;
};

const STORAGE_KEY = "admin-use-cases";

const initialUseCaseData: UseCaseItem[] = [
  {
    id: "10021350",
    title: "Use Case Title 01",
    category: "Category 1",
    description:
      "Placeholder description for this use case. Real content can be added later.",
  },
  {
    id: "10021351",
    title: "Use Case Title 02",
    category: "Category 2",
    description:
      "Placeholder description for this use case. Real content can be added later.",
  },
  {
    id: "10021352",
    title: "Use Case Title 03",
    category: "Category 1",
    description:
      "Placeholder description for this use case. Real content can be added later.",
  },
  {
    id: "10021353",
    title: "Use Case Title 04",
    category: "Category 3",
    description:
      "Placeholder description for this use case. Real content can be added later.",
  },
  {
    id: "10021354",
    title: "Use Case Title 05",
    category: "Category 4",
    description:
      "Placeholder description for this use case. Real content can be added later.",
  },
];

const categoryOptions = ["Category 1", "Category 2", "Category 3", "Category 4"];

export default function AddNewUseCasePage() {
  const router = useRouter();
  const params = useParams();

  const localeParam = params.locale;
  const locale = Array.isArray(localeParam)
    ? localeParam[0]
    : localeParam || "en";

  const [title, setTitle] = useState("");
  const [category, setCategory] = useState("Category 1");
  const [description, setDescription] = useState("");

  const handleBack = () => {
    router.push(`/${locale}/admin/use-cases`);
  };

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const storedData = localStorage.getItem(STORAGE_KEY);
    let currentData: UseCaseItem[] = initialUseCaseData;

    if (storedData) {
      try {
        currentData = JSON.parse(storedData);
      } catch {
        currentData = initialUseCaseData;
      }
    }

    const numericIds = currentData
      .map((item) => Number(item.id))
      .filter((id) => Number.isFinite(id));

    const nextId = String(
      (numericIds.length > 0 ? Math.max(...numericIds) : 10021349) + 1
    );

    const newUseCase: UseCaseItem = {
      id: nextId,
      title: title.trim(),
      category,
      description: description.trim(),
    };

    const nextData = [newUseCase, ...currentData];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(nextData));

    router.push(`/${locale}/admin/use-cases`);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-[40px] font-semibold leading-[48px] text-[#2DBE6C]">
          Add New Use Case
        </h1>
        <p className="mt-1 text-sm text-[#667085]">
          Create a new use case entry
        </p>
      </div>

      {/* Form */}
    <form
        onSubmit={handleSubmit}
        className="w-full rounded-2xl border border-[#E4E7EC] bg-white p-6 shadow-sm"
    >
        <div className="space-y-6">
          <div>
            <label className="mb-2 block text-sm font-medium text-[#344054]">
              Use Case Title
            </label>
            <input
              type="text"
              placeholder="Enter use case title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              required
              className="h-11 w-full rounded-md border border-[#2DBE6C] bg-white px-4 text-sm text-[#344054] outline-none transition focus:ring-1 focus:ring-[#2DBE6C]"
            />
          </div>

          <div>
            <label className="mb-2 block text-sm font-medium text-[#344054]">
              Category
            </label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="h-11 w-full rounded-md border border-[#2DBE6C] bg-white px-4 text-sm text-[#344054] outline-none transition focus:ring-1 focus:ring-[#2DBE6C]"
            >
              {categoryOptions.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="mb-2 block text-sm font-medium text-[#344054]">
              Description
            </label>
            <textarea
              placeholder="Enter use case description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={6}
              required
              className="w-full resize-none rounded-md border border-[#2DBE6C] bg-white px-4 py-3 text-sm text-[#344054] outline-none transition focus:ring-1 focus:ring-[#2DBE6C]"
            />
          </div>

          <div>
            <label className="mb-2 block text-sm font-medium text-[#344054]">
              Cover Image
            </label>
            <div className="flex h-[160px] flex-col items-center justify-center rounded-md border border-dashed border-[#2DBE6C] bg-[#F9FAFB] px-4 text-center">
              <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-[#E8F8EE] text-[#2DBE6C]">
                <ImagePlus size={22} />
              </div>
              <p className="text-sm font-medium text-[#344054]">
                Upload cover image
              </p>
              <p className="mt-1 text-xs text-[#667085]">
                Placeholder upload area
              </p>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="mt-8 flex gap-4">
          <button
            type="button"
            onClick={handleBack}
            className="inline-flex h-10 min-w-[90px] items-center justify-center gap-2 rounded-md border border-[#2DBE6C] bg-white px-5 text-sm font-medium text-[#2DBE6C] transition hover:bg-[#F6FEF9]"
          >
            <ArrowLeft size={16} />
            Back
          </button>

          <button
            type="submit"
            className="inline-flex h-10 min-w-[90px] items-center justify-center gap-2 rounded-md bg-[#188C45] px-5 text-sm font-medium text-white transition hover:bg-[#147a3c]"
          >
            <Save size={16} />
            Save
          </button>
        </div>
      </form>
    </div>
  );
}