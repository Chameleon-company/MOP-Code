"use client";

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import SearchFilter from "../../../components/SearchFilter";
import SearchResults from "../../../components/SearchResults";

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
    const [error, setError] = useState(false);

    const performSearch = async () => {
        setLoading(true);
        setError(false);
        try {
            const params = new URLSearchParams(searchParams.toString());
            const response = await fetch(`/api/search?${params.toString()}`);
            const data = await response.json();
            if (data.success) {
                setSearchResults(data.data);
            } else {
                setError(true);
            }
        } catch {
            setError(true);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        performSearch();
    }, [searchParams]);

    const handleFilterChange = () => {
        performSearch();
    };

    const goToPage = (newPage: number) => {
        const newParams = new URLSearchParams(searchParams.toString());
        newParams.set("page", String(newPage));
        window.history.pushState({}, "", `?${newParams.toString()}`);
        performSearch();
    };

    return (
        <div className="flex min-h-screen flex-col bg-white dark:bg-[#1C1C1C] text-gray-900 dark:text-white">
            <Header />

            <main className="flex-1 mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold">Search</h1>
                    {searchResults?.filters?.query && (
                        <p className="mt-2 text-gray-500 dark:text-gray-400">
                            Results for &ldquo;{searchResults.filters.query}&rdquo;
                        </p>
                    )}
                </div>

                <div className="grid grid-cols-1 gap-8 lg:grid-cols-4">
                    {/* Filters Sidebar */}
                    <div className="lg:col-span-1">
                        <SearchFilter onFilterChange={handleFilterChange} />
                    </div>

                    {/* Results */}
                    <div className="lg:col-span-3">
                        {loading ? (
                            <div className="flex items-center justify-center py-20">
                                <div className="h-10 w-10 animate-spin rounded-full border-4 border-green-200 border-t-green-600" />
                            </div>
                        ) : error ? (
                            <div className="flex min-h-[200px] items-center justify-center rounded-2xl border border-dashed border-red-200 bg-red-50 px-6 py-12 text-center dark:border-red-900 dark:bg-red-950/20">
                                <p className="text-base font-medium text-red-500 dark:text-red-400">
                                    Something went wrong. Please try again later.
                                </p>
                            </div>
                        ) : searchResults ? (
                            <div>
                                <div className="mb-6">
                                    <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                                        {searchResults.pagination.total} Result{searchResults.pagination.total !== 1 ? "s" : ""}
                                    </h2>
                                </div>

                                {searchResults.results.length === 0 ? (
                                    <div className="flex min-h-[200px] items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-gray-50 px-6 py-12 text-center dark:border-gray-700 dark:bg-[#242424]">
                                        <p className="text-base font-medium text-gray-500 dark:text-gray-400">
                                            No results found. Try a different search term.
                                        </p>
                                    </div>
                                ) : (
                                    <SearchResults
                                        results={searchResults.results}
                                        loading={loading}
                                        pagination={searchResults.pagination}
                                    />
                                )}

                                {searchResults.pagination.totalPages > 1 && (
                                    <div className="mt-8 flex items-center justify-center gap-3">
                                        {searchResults.pagination.hasPrev && (
                                            <button
                                                onClick={() => goToPage(searchResults.pagination.page - 1)}
                                                className="inline-flex items-center justify-center rounded-xl border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors dark:border-gray-700 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700"
                                            >
                                                Previous
                                            </button>
                                        )}
                                        <span className="rounded-xl bg-green-600 px-4 py-2 text-sm font-semibold text-white">
                                            {searchResults.pagination.page} / {searchResults.pagination.totalPages}
                                        </span>
                                        {searchResults.pagination.hasNext && (
                                            <button
                                                onClick={() => goToPage(searchResults.pagination.page + 1)}
                                                className="inline-flex items-center justify-center rounded-xl border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors dark:border-gray-700 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700"
                                            >
                                                Next
                                            </button>
                                        )}
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="flex min-h-[200px] items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-gray-50 px-6 py-12 text-center dark:border-gray-700 dark:bg-[#242424]">
                                <p className="text-base font-medium text-gray-500 dark:text-gray-400">
                                    Enter search terms to find results.
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            </main>

            <Footer />
        </div>
    );
}