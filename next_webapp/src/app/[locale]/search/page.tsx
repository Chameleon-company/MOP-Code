'use client';

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import SearchFilter from '../../../components/SearchFilter';
import SearchResults from '../../../components/SearchResults';

interface SearchResult {
    id: number;
    title: string;
    description?: string;
    content?: string;
    category_name?: string;
    category_slug?: string;
    category_color?: string;
    category_icon?: string;
    created_at: string;
    updated_at?: string;
    view_count?: number;
}

interface Pagination {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
}

interface SearchData {
    results: SearchResult[];
    pagination: Pagination;
    filters: {
        query: string;
        category?: string;
        categories: string[];
        sortBy: string;
        sortOrder: string;
    };
}

export default function SearchPage() {
    const searchParams = useSearchParams();
    const [searchResults, setSearchResults] = useState<SearchData | null>(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const performSearch = async () => {
            setLoading(true);
            try {
                const params = new URLSearchParams(searchParams.toString());
                const response = await fetch(`/api/search?${params.toString()}`);
                const data = await response.json();
                
                if (data.success) {
                    setSearchResults(data.data);
                }
            } catch (error) {
                console.error('Search error:', error);
            } finally {
                setLoading(false);
            }
        };

        performSearch();
    }, [searchParams]);

    const handleRefresh = async () => {
        setLoading(true);
        try {
            const params = new URLSearchParams(searchParams.toString());
            const response = await fetch(`/api/search?${params.toString()}`);
            const data = await response.json();
            
            if (data.success) {
                setSearchResults(data.data);
            }
        } catch (error) {
            console.error('Search error:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleFilterChange = () => {
        // This will trigger the useEffect above through URL change
        handleRefresh();
    };

    return (
        <div className="container mx-auto px-4 py-8">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
                {/* Filters Sidebar */}
                <div className="lg:col-span-1">
                    <SearchFilter onFilterChange={handleFilterChange} />
                </div>

                {/* Search Results */}
                <div className="lg:col-span-3">
                    {loading ? (
                        <div className="text-center py-8">Loading...</div>
                    ) : searchResults ? (
                        <div>
                            <div className="mb-6">
                                <h2 className="text-2xl font-bold">
                                    Search Results ({searchResults.pagination.total})
                                </h2>
                                {searchResults.filters.query && (
                                    <p className="text-gray-600 mt-2">
                                        Results for &quot;{searchResults.filters.query}&quot;
                                    </p>
                                )}
                            </div>

                            {/* Results List */}
                            <SearchResults 
                                results={searchResults.results} 
                                loading={loading}
                                pagination={searchResults.pagination}
                            />

                            {/* Pagination */}
                            {searchResults.pagination.totalPages > 1 && (
                                <div className="mt-8 flex justify-center">
                                    <div className="flex space-x-2">
                                        {searchResults.pagination.hasPrev && (
                                            <button 
                                                className="px-4 py-2 border rounded hover:bg-gray-50"
                                                onClick={() => {
                                                    const newParams = new URLSearchParams(searchParams.toString());
                                                    newParams.set('page', String(searchResults.pagination.page - 1));
                                                    window.history.pushState({}, '', `?${newParams.toString()}`);
                                                    handleRefresh();
                                                }}
                                            >
                                                Previous
                                            </button>
                                        )}
                                        
                                        <span className="px-4 py-2 bg-blue-500 text-white rounded">
                                            {searchResults.pagination.page} of {searchResults.pagination.totalPages}
                                        </span>
                                        
                                        {searchResults.pagination.hasNext && (
                                            <button 
                                                className="px-4 py-2 border rounded hover:bg-gray-50"
                                                onClick={() => {
                                                    const newParams = new URLSearchParams(searchParams.toString());
                                                    newParams.set('page', String(searchResults.pagination.page + 1));
                                                    window.history.pushState({}, '', `?${newParams.toString()}`);
                                                    handleRefresh();
                                                }}
                                            >
                                                Next
                                            </button>
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="text-center py-8">
                            <p>Enter search terms to find results</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
