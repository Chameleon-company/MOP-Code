'use client';

import { useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

interface Category {
    id: number;
    name: string;
    slug: string;
    color: string;
    icon?: string;
    description?: string;
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
    const searchParams = useSearchParams();
    
    const [categories, setCategories] = useState<Category[]>([]);
    const [filters, setFilters] = useState<FilterState>({
        query: searchParams.get('q') || '',
        selectedCategories: searchParams.getAll('categories') || [],
        sortBy: searchParams.get('sortBy') || 'created_at',
        sortOrder: searchParams.get('sortOrder') || 'DESC'
    });

    // Fetch categories on component mount
    useEffect(() => {
        fetchCategories();
    }, []);

    const fetchCategories = async () => {
        try {
            const response = await fetch('/api/categories');
            const data = await response.json();
            if (data.success) {
                setCategories(data.data);
            }
        } catch (error) {
            console.error('Error fetching categories:', error);
        }
    };

    const handleFilterChange = (newFilters: FilterState) => {
        setFilters(newFilters);
        
        // Update URL params
        const params = new URLSearchParams();
        
        if (newFilters.query) params.set('q', newFilters.query);
        if (newFilters.sortBy) params.set('sortBy', newFilters.sortBy);
        if (newFilters.sortOrder) params.set('sortOrder', newFilters.sortOrder);
        
        newFilters.selectedCategories.forEach((cat: string) => {
            params.append('categories', cat);
        });

        router.push(`/search?${params.toString()}`);
        
        // Notify parent component
        if (onFilterChange) {
            onFilterChange(newFilters);
        }
    };

    const handleCategoryToggle = (categorySlug: string) => {
        const updatedCategories = filters.selectedCategories.includes(categorySlug)
            ? filters.selectedCategories.filter(cat => cat !== categorySlug)
            : [...filters.selectedCategories, categorySlug];

        handleFilterChange({
            ...filters,
            selectedCategories: updatedCategories
        });
    };

    const clearFilters = () => {
        const clearedFilters = {
            query: '',
            selectedCategories: [],
            sortBy: 'created_at',
            sortOrder: 'DESC'
        };
        handleFilterChange(clearedFilters);
    };

    return (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-4">Search Filters</h3>
            
            {/* Search Input */}
            <div className="mb-6">
                <label className="block text-sm font-medium mb-2">Search Query</label>
                <input
                    type="text"
                    value={filters.query}
                    onChange={(e) => handleFilterChange({ ...filters, query: e.target.value })}
                    placeholder="Enter search terms..."
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
            </div>

            {/* Categories Filter */}
            <div className="mb-6">
                <label className="block text-sm font-medium mb-3">Categories</label>
                <div className="space-y-2">
                    {categories.map((category) => (
                        <label key={category.id} className="flex items-center">
                            <input
                                type="checkbox"
                                checked={filters.selectedCategories.includes(category.slug)}
                                onChange={() => handleCategoryToggle(category.slug)}
                                className="mr-2"
                            />
                            <span 
                                className="inline-block w-3 h-3 rounded-full mr-2"
                                style={{ backgroundColor: category.color }}
                            ></span>
                            <span>{category.name}</span>
                        </label>
                    ))}
                </div>
            </div>

            {/* Sort Options */}
            <div className="mb-6">
                <label className="block text-sm font-medium mb-2">Sort By</label>
                <select
                    value={`${filters.sortBy}-${filters.sortOrder}`}
                    onChange={(e) => {
                        const [sortBy, sortOrder] = e.target.value.split('-');
                        handleFilterChange({ ...filters, sortBy, sortOrder });
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

            {/* Clear Filters */}
            <button
                onClick={clearFilters}
                className="w-full px-4 py-2 text-sm text-gray-600 hover:text-gray-800 border border-gray-300 rounded-md hover:bg-gray-50"
            >
                Clear All Filters
            </button>
        </div>
    );
}
