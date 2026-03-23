'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import SearchResults from '../../../../components/SearchResults';

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

interface Category {
    name: string;
    color: string;
    description?: string;
}

export default function CategoryPage() {
    const params = useParams();
    const [category, setCategory] = useState<Category | null>(null);
    const [posts, setPosts] = useState<SearchResult[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (params.slug) {
            const fetchCategoryPosts = async () => {
                try {
                    // Fetch category info and posts
                    const response = await fetch(`/api/search?category=${params.slug}`);
                    const data = await response.json();
                    
                    if (data.success) {
                        setPosts(data.data.results);
                        if (data.data.results.length > 0) {
                            setCategory({
                                name: data.data.results[0].category_name || 'Unknown Category',
                                color: data.data.results[0].category_color || '#6B7280',
                                description: data.data.results[0].description
                            });
                        }
                    }
                } catch (error) {
                    console.error('Error fetching category posts:', error);
                } finally {
                    setLoading(false);
                }
            };
            
            fetchCategoryPosts();
        }
    }, [params.slug]);

    return (
        <div className="container mx-auto px-4 py-8">
            {category && (
                <div className="mb-8">
                    <div className="flex items-center mb-4">
                        <span 
                            className="inline-block w-6 h-6 rounded-full mr-3"
                            style={{ backgroundColor: category.color }}
                        ></span>
                        <h1 className="text-3xl font-bold">{category.name}</h1>
                    </div>
                    {category.description && (
                        <p className="text-gray-600 text-lg">{category.description}</p>
                    )}
                </div>
            )}
            
            <SearchResults results={posts} loading={loading} />
        </div>
    );
}
