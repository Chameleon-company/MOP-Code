'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

interface Category {
    id: number;
    name: string;
    slug: string;
    color: string;
    icon?: string;
    description?: string;
}

export default function CategoriesPage() {
    const [categories, setCategories] = useState<Category[]>([]);
    const [loading, setLoading] = useState(true);

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
        } finally {
            setLoading(false);
        }
    };

    if (loading) return <div className="text-center py-8">Loading...</div>;

    return (
        <div className="container mx-auto px-4 py-8">
            <h1 className="text-3xl font-bold mb-8">All Categories</h1>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {categories.map((category) => (
                    <Link key={category.id} href={`/categories/${category.slug}`}>
                        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow cursor-pointer">
                            <div className="flex items-center mb-3">
                                <span 
                                    className="inline-block w-4 h-4 rounded-full mr-3"
                                    style={{ backgroundColor: category.color }}
                                ></span>
                                <h2 className="text-xl font-semibold">{category.name}</h2>
                            </div>
                            {category.description && (
                                <p className="text-gray-600">{category.description}</p>
                            )}
                        </div>
                    </Link>
                ))}
            </div>
        </div>
    );
}
