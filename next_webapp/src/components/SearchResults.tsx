'use client';

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

interface SearchResultsProps {
    results: SearchResult[];
    loading: boolean;
    pagination?: Pagination;
}

export default function SearchResults({ results, loading, pagination }: SearchResultsProps) {
    if (loading) {
        return (
            <div className="flex justify-center items-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            </div>
        );
    }

    if (!results || results.length === 0) {
        return (
            <div className="text-center py-8">
                <p className="text-gray-500">No results found</p>
            </div>
        );
    }

    return (
        <div className="search-results">
            <div className="mb-4">
                <p className="text-sm text-gray-600">
                    Showing {results.length} of {pagination?.total || results.length} results
                </p>
            </div>
            
            <div className="space-y-4">
                {results.map((result, index) => (
                    <div key={result.id || index} className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm border">
                        <div className="flex items-center mb-2">
                            {result.category_name && (
                                <span 
                                    className="inline-block px-2 py-1 text-xs font-semibold rounded-full text-white mr-2"
                                    style={{ backgroundColor: result.category_color || '#6B7280' }}
                                >
                                    {result.category_name}
                                </span>
                            )}
                        </div>
                        
                        <h3 className="text-lg font-semibold mb-2">
                            <a 
                                href={`/posts/${result.id}`} 
                                className="hover:text-blue-600 transition-colors"
                            >
                                {result.title}
                            </a>
                        </h3>
                        
                        {result.description && (
                            <p className="text-gray-600 mb-3 line-clamp-2">
                                {result.description}
                            </p>
                        )}
                        
                        <div className="text-xs text-gray-500 flex justify-between items-center">
                            <span>
                                {result.created_at && new Date(result.created_at).toLocaleDateString()}
                            </span>
                            {result.view_count && (
                                <span>
                                    {result.view_count} views
                                </span>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
