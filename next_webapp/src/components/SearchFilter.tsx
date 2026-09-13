'use client';

import { useState, useEffect } from 'react';
import { useRouter, useSearchParams, usePathname } from 'next/navigation';
import { apiFetch } from '@/lib/apiFetch';

// Matches the Mongoose Category schema (src/models/mongoose/Category.ts).
// No slug, color, or icon fields exist on the schema.
interface Category {
    _id: string;
    legacy_id: string | null;
    category_name: string;
    description?: string | null;
    cover_img?: string | null;
}

interface FilterState {
    query: string;
    selectedCategories: string[];
    sortBy: string;
    sortOrder: string;
}

interface SearchFilterProps {
    onFilterChange?: (filters: FilterState) => void;
}

export default function SearchFilter({ onFilterChange }: SearchFilterProps) {
    const router = useRouter();
    const pathname = usePathname();
    const searchParams = useSearchParams();

    const [categories, setCategories] = useState<Category[]>([]);

    // All filter inputs are held in local (draft) state.
    // Nothing is pushed to the URL until the user clicks "Apply Filters".
    const [draft, setDraft] = useState<FilterState>({
        query: searchParams.get('q') || '',
        selectedCategories: searchParams.getAll('categories') || [],
        sortBy: searchParams.get('sortBy') || 'created_at',
        sortOrder: searchParams.get('sortOrder') || 'DESC',
    });

    // Fetch categories on component mount
    useEffect(() => {
        fetchCategories();
    }, []);

    const fetchCategories = async () => {
        try {
            const data = await apiFetch<{ success: boolean; data: Category[] }>('/api/categories');
            if (data.success) {
                setCategories(data.data);
            }
        } catch (error) {
            console.error('Error fetching categories:', error); // apiFetch already showed a toast
        }
    };

    // Commit all draft filters to the URL at once.
    const applyFilters = () => {
        const params = new URLSearchParams();

        if (draft.query.trim()) params.set('q', draft.query.trim());
        if (draft.sortBy) params.set('sortBy', draft.sortBy);
        if (draft.sortOrder) params.set('sortOrder', draft.sortOrder);

        draft.selectedCategories.forEach((cat: string) => {
            params.append('categories', cat);
        });

        // Use pathname so the locale prefix (/en/, /fr/, etc.) is preserved.
        router.push(`${pathname}?${params.toString()}`);

        if (onFilterChange) {
            onFilterChange(draft);
        }
    };

    const handleCategoryToggle = (categoryId: string) => {
        // Only update local draft — not pushed to URL yet.
        // Use legacy_id (falling back to _id) as the stable identifier.
        const updated = draft.selectedCategories.includes(categoryId)
            ? draft.selectedCategories.filter(cat => cat !== categoryId)
            : [...draft.selectedCategories, categoryId];

        setDraft(prev => ({ ...prev, selectedCategories: updated }));
    };

    const clearFilters = () => {
        const cleared: FilterState = {
            query: '',
            selectedCategories: [],
            sortBy: 'created_at',
            sortOrder: 'DESC',
        };
        setDraft(cleared);

        // Also clear the URL immediately.
        router.push(pathname);

        if (onFilterChange) {
            onFilterChange(cleared);
        }
    };

    return (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-4">Search Filters</h3>

            {/* Search Query */}
            <div className="mb-6">
                <label htmlFor="search-query" className="block text-sm font-medium mb-2">Search Query</label>
                <input
                    id="search-query"
                    type="text"
                    value={draft.query}
                    onChange={(e) => setDraft(prev => ({ ...prev, query: e.target.value }))}
                    onKeyDown={(e) => e.key === 'Enter' && applyFilters()}
                    placeholder="Enter search terms..."
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
            </div>

            {/* Categories Filter */}
            <div className="mb-6">
                <label className="block text-sm font-medium mb-3">Categories</label>
                <div className="space-y-2">
                    {categories.map((category) => {
                        // Use legacy_id when available (stable Postgres-era ID),
                        // fall back to _id (MongoDB ObjectId string).
                        const categoryId = category.legacy_id ?? category._id;
                        return (
                            <label key={category._id} className="flex items-center cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={draft.selectedCategories.includes(categoryId)}
                                    onChange={() => handleCategoryToggle(categoryId)}
                                    className="mr-2"
                                />
                                <span>{category.category_name}</span>
                            </label>
                        );
                    })}
                </div>
            </div>

            {/* Sort Options */}
            <div className="mb-6">
                <label htmlFor="sort-by" className="block text-sm font-medium mb-2">Sort By</label>
                <select
                    id="sort-by"
                    value={`${draft.sortBy}-${draft.sortOrder}`}
                    onChange={(e) => {
                        const [sortBy, sortOrder] = e.target.value.split('-');
                        setDraft(prev => ({ ...prev, sortBy, sortOrder }));
                    }}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                    <option value="created_at-DESC">Newest First</option>
                    <option value="created_at-ASC">Oldest First</option>
                    <option value="title-ASC">Title A-Z</option>
                    <option value="title-DESC">Title Z-A</option>
                    <option value="view_count-DESC">Most Popular</option>
                </select>
            </div>

            {/* Shared action buttons */}
            <button
                onClick={applyFilters}
                className="w-full px-4 py-2 mb-2 text-sm text-white bg-green-600 hover:bg-green-700 rounded-md font-medium"
            >
                Apply Filters
            </button>
            <button
                onClick={clearFilters}
                className="w-full px-4 py-2 text-sm text-gray-600 hover:text-gray-800 border border-gray-300 rounded-md hover:bg-gray-50"
            >
                Clear All Filters
            </button>
        </div>
    );
}
