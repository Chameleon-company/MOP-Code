"use client";

import { useMemo, useState, type FormEvent } from "react";
import {
  Search,
  Filter,
  Pencil,
  Trash2,
  Plus,
  ChevronLeft,
  ChevronRight,
  X,
  Save,
  ImagePlus,
} from "lucide-react";

type UseCaseItem = {
  id: string;
  title: string;
  category: string;
  description: string;
};

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
  {
    id: "10021355",
    title: "Use Case Title 06",
    category: "Category 2",
    description:
      "Placeholder description for this use case. Real content can be added later.",
  },
  {
    id: "10021356",
    title: "Use Case Title 07",
    category: "Category 3",
    description:
      "Placeholder description for this use case. Real content can be added later.",
  },
  {
    id: "10021357",
    title: "Use Case Title 08",
    category: "Category 1",
    description:
      "Placeholder description for this use case. Real content can be added later.",
  },
];

const categoryOptions = ["Category 1", "Category 2", "Category 3", "Category 4"];

const ITEMS_PER_PAGE = 5;

export default function UseCasesPage() {
  const [useCases, setUseCases] = useState<UseCaseItem[]>(initialUseCaseData);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [page, setPage] = useState(1);

  const [showAddModal, setShowAddModal] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [newCategory, setNewCategory] = useState("Category 1");
  const [newDescription, setNewDescription] = useState("");

  const filteredUseCases = useMemo(() => {
    const keyword = searchTerm.trim().toLowerCase();

    if (!keyword) return useCases;

    return useCases.filter((item) => {
      return (
        item.id.toLowerCase().includes(keyword) ||
        item.title.toLowerCase().includes(keyword) ||
        item.category.toLowerCase().includes(keyword) ||
        item.description.toLowerCase().includes(keyword)
      );
    });
  }, [searchTerm, useCases]);

  const totalPages = Math.max(
    1,
    Math.ceil(filteredUseCases.length / ITEMS_PER_PAGE)
  );

  const currentPage = Math.min(page, totalPages);

  const paginatedUseCases = filteredUseCases.slice(
    (currentPage - 1) * ITEMS_PER_PAGE,
    currentPage * ITEMS_PER_PAGE
  );

  const allCurrentPageSelected =
    paginatedUseCases.length > 0 &&
    paginatedUseCases.every((item) => selectedIds.includes(item.id));

  const toggleSelectAll = () => {
    if (allCurrentPageSelected) {
      setSelectedIds((prev) =>
        prev.filter((id) => !paginatedUseCases.some((item) => item.id === id))
      );
    } else {
      const newIds = paginatedUseCases.map((item) => item.id);
      setSelectedIds((prev) => Array.from(new Set([...prev, ...newIds])));
    }
  };

  const toggleSelectOne = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const handleSearchChange = (value: string) => {
    setSearchTerm(value);
    setPage(1);
  };

  const resetAddForm = () => {
    setNewTitle("");
    setNewCategory("Category 1");
    setNewDescription("");
  };

  const closeAddModal = () => {
    resetAddForm();
    setShowAddModal(false);
  };

  const handleAddUseCase = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const numericIds = useCases
      .map((item) => Number(item.id))
      .filter((id) => Number.isFinite(id));

    const nextId = String(
      (numericIds.length > 0 ? Math.max(...numericIds) : 10021349) + 1
    );

    const newUseCase: UseCaseItem = {
      id: nextId,
      title: newTitle.trim(),
      category: newCategory,
      description: newDescription.trim(),
    };

    setUseCases((prev) => [newUseCase, ...prev]);
    setSearchTerm("");
    setPage(1);
    resetAddForm();
    setShowAddModal(false);
  };

  const handleDeleteUseCase = (id: string) => {
    setUseCases((prev) => prev.filter((item) => item.id !== id));
    setSelectedIds((prev) => prev.filter((selectedId) => selectedId !== id));
  };

  return (
    <>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <h1 className="text-[40px] font-semibold leading-[48px] text-[#2DBE6C]">
              Use Cases
            </h1>
            <p className="mt-1 text-sm text-[#667085]">
              Manage and organize your use cases
            </p>
          </div>

          <button
            type="button"
            onClick={() => setShowAddModal(true)}
            className="inline-flex w-fit items-center gap-2 rounded-md bg-[#2DBE6C] px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-[#25a85e]"
          >
            <Plus size={16} />
            Add New
          </button>
        </div>

        {/* Search */}
        <div className="rounded-2xl border border-[#E4E7EC] bg-white p-4 shadow-sm">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div className="relative w-full md:max-w-[420px]">
              <Search
                size={18}
                className="absolute left-4 top-1/2 -translate-y-1/2 text-[#98A2B3]"
              />
              <input
                type="text"
                placeholder="Search use cases..."
                value={searchTerm}
                onChange={(e) => handleSearchChange(e.target.value)}
                className="h-11 w-full rounded-xl border border-[#E4E7EC] bg-[#F9FAFB] pl-11 pr-4 text-sm text-[#344054] outline-none transition focus:border-[#2DBE6C] focus:bg-white"
              />
            </div>

            <button
              type="button"
              className="inline-flex h-11 w-11 items-center justify-center rounded-xl border border-[#E4E7EC] bg-[#F9FAFB] text-[#667085] transition hover:bg-[#F2F4F7]"
            >
              <Filter size={16} />
            </button>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-hidden rounded-2xl border border-[#E4E7EC] bg-white shadow-sm">
          <div className="overflow-x-auto">
            <table className="min-w-full table-fixed">
              <thead className="bg-[#F9FAFB]">
                <tr className="text-left text-xs font-medium text-[#667085]">
                  <th className="w-[56px] px-4 py-3">
                    <input
                      type="checkbox"
                      checked={allCurrentPageSelected}
                      onChange={toggleSelectAll}
                      className="h-4 w-4 rounded border-[#D0D5DD] accent-[#2DBE6C] focus:ring-[#2DBE6C]"
                    />
                  </th>
                  <th className="w-[120px] px-4 py-3">Serial Number</th>
                  <th className="w-[180px] px-4 py-3">Title</th>
                  <th className="w-[120px] px-4 py-3">Cover Image</th>
                  <th className="w-[130px] px-4 py-3">Category</th>
                  <th className="px-4 py-3">Description</th>
                  <th className="w-[110px] px-4 py-3">Action</th>
                </tr>
              </thead>

              <tbody>
                {paginatedUseCases.length > 0 ? (
                  paginatedUseCases.map((item) => (
                    <tr
                      key={item.id}
                      className="border-t border-[#EAECF0] text-sm text-[#344054]"
                    >
                      <td className="px-4 py-4 align-top">
                        <input
                          type="checkbox"
                          checked={selectedIds.includes(item.id)}
                          onChange={() => toggleSelectOne(item.id)}
                          className="h-4 w-4 rounded border-[#D0D5DD] accent-[#2DBE6C] focus:ring-[#2DBE6C]"
                        />
                      </td>

                      <td className="px-4 py-4 align-top text-[#667085]">
                        {item.id}
                      </td>

                      <td className="px-4 py-4 align-top font-medium text-[#344054]">
                        {item.title}
                      </td>

                      <td className="px-4 py-4 align-top">
                        <div className="h-10 w-10 rounded bg-[#D0D5DD]" />
                      </td>

                      <td className="px-4 py-4 align-top">
                        <span className="inline-flex rounded-full bg-[#F2F4F7] px-3 py-1 text-xs font-medium text-[#344054]">
                          {item.category}
                        </span>
                      </td>

                      <td className="px-4 py-4 align-top text-[#667085]">
                        <p className="line-clamp-2 max-w-[420px]">
                          {item.description}
                        </p>
                      </td>

                      <td className="px-4 py-4 align-top">
                        <div className="flex items-center gap-3">
                          <button
                            type="button"
                            className="text-[#2DBE6C] transition hover:text-[#25a85e]"
                          >
                            <Pencil size={16} />
                          </button>
                          <button
                            type="button"
                            onClick={() => handleDeleteUseCase(item.id)}
                            className="text-[#F04438] transition hover:text-[#d92d20]"
                          >
                            <Trash2 size={16} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={7} className="px-6 py-16 text-center">
                      <div className="space-y-2">
                        <p className="text-base font-medium text-[#101828]">
                          No use cases found
                        </p>
                        <p className="text-sm text-[#667085]">
                          Try another keyword or add a new use case.
                        </p>
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex flex-col gap-3 border-t border-[#EAECF0] px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-sm text-[#667085]">
              Showing{" "}
              <span className="font-medium text-[#344054]">
                {filteredUseCases.length}
              </span>{" "}
              use case{filteredUseCases.length !== 1 ? "s" : ""}
            </p>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setPage((prev) => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
                className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-[#D0D5DD] text-[#667085] transition hover:bg-[#F9FAFB] disabled:cursor-not-allowed disabled:opacity-50"
              >
                <ChevronLeft size={16} />
              </button>

              {Array.from({ length: totalPages }).map((_, index) => {
                const pageNumber = index + 1;
                const active = pageNumber === currentPage;

                return (
                  <button
                    key={pageNumber}
                    type="button"
                    onClick={() => setPage(pageNumber)}
                    className={`inline-flex h-8 w-8 items-center justify-center rounded-md text-sm font-medium transition ${
                      active
                        ? "bg-[#2DBE6C] text-white"
                        : "border border-[#D0D5DD] text-[#667085] hover:bg-[#F9FAFB]"
                    }`}
                  >
                    {pageNumber}
                  </button>
                );
              })}

              <button
                type="button"
                onClick={() =>
                  setPage((prev) => Math.min(prev + 1, totalPages))
                }
                disabled={currentPage === totalPages}
                className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-[#D0D5DD] text-[#667085] transition hover:bg-[#F9FAFB] disabled:cursor-not-allowed disabled:opacity-50"
              >
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Add New Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4">
          <div className="w-full max-w-2xl overflow-hidden rounded-2xl bg-white shadow-xl">
            <div className="flex items-start justify-between border-b border-[#EAECF0] px-6 py-5">
              <div>
                <h2 className="text-xl font-semibold text-[#101828]">
                  Add New Use Case
                </h2>
                <p className="mt-1 text-sm text-[#667085]">
                  Add placeholder information for a new use case.
                </p>
              </div>

              <button
                type="button"
                onClick={closeAddModal}
                className="rounded-full p-1 text-[#667085] transition hover:bg-[#F2F4F7] hover:text-[#101828]"
              >
                <X size={20} />
              </button>
            </div>

            <form onSubmit={handleAddUseCase} className="px-6 py-5">
              <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
                <div className="space-y-5 md:col-span-2">
                  <div>
                    <label className="mb-2 block text-sm font-medium text-[#344054]">
                      Title
                    </label>
                    <input
                      type="text"
                      value={newTitle}
                      onChange={(e) => setNewTitle(e.target.value)}
                      placeholder="Enter use case title"
                      required
                      className="h-11 w-full rounded-xl border border-[#D0D5DD] px-4 text-sm text-[#344054] outline-none transition focus:border-[#2DBE6C]"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm font-medium text-[#344054]">
                      Category
                    </label>
                    <select
                      value={newCategory}
                      onChange={(e) => setNewCategory(e.target.value)}
                      className="h-11 w-full rounded-xl border border-[#D0D5DD] px-4 text-sm text-[#344054] outline-none transition focus:border-[#2DBE6C]"
                    >
                      {categoryOptions.map((category) => (
                        <option key={category} value={category}>
                          {category}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="mb-2 block text-sm font-medium text-[#344054]">
                      Description
                    </label>
                    <textarea
                      value={newDescription}
                      onChange={(e) => setNewDescription(e.target.value)}
                      placeholder="Enter use case description"
                      rows={5}
                      required
                      className="w-full resize-none rounded-xl border border-[#D0D5DD] px-4 py-3 text-sm text-[#344054] outline-none transition focus:border-[#2DBE6C]"
                    />
                  </div>
                </div>

                <div>
                  <label className="mb-2 block text-sm font-medium text-[#344054]">
                    Cover Image
                  </label>
                  <div className="flex h-[220px] flex-col items-center justify-center rounded-2xl border border-dashed border-[#D0D5DD] bg-[#F9FAFB] px-4 text-center">
                    <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-[#E8F8EE] text-[#2DBE6C]">
                      <ImagePlus size={22} />
                    </div>
                    <p className="text-sm font-medium text-[#344054]">
                      Upload image
                    </p>
                    <p className="mt-1 text-xs text-[#667085]">
                      Placeholder only
                    </p>
                  </div>
                </div>
              </div>

              <div className="mt-6 flex flex-col-reverse gap-3 border-t border-[#EAECF0] pt-5 sm:flex-row sm:justify-end">
                <button
                  type="button"
                  onClick={closeAddModal}
                  className="inline-flex h-10 items-center justify-center rounded-md border border-[#D0D5DD] bg-white px-5 text-sm font-medium text-[#344054] transition hover:bg-[#F9FAFB]"
                >
                  Cancel
                </button>

                <button
                  type="submit"
                  className="inline-flex h-10 items-center justify-center gap-2 rounded-md bg-[#2DBE6C] px-5 text-sm font-medium text-white shadow-sm transition hover:bg-[#25a85e]"
                >
                  <Save size={16} />
                  Save
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}