'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import Header from '@/components/Header';
import Footer from '@/components/Footer';

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
    const [error, setError] = useState(false);

    useEffect(() => {
        fetchCategories();
    }, []);

    const fetchCategories = async () => {
        try {
            setError(false);
            const response = await fetch('/api/categories');
            const data = await response.json();
            if (data.success) {
                setCategories(data.data);
            } else {
                setError(true);
            }
        } catch {
            setError(true);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex min-h-screen flex-col bg-white text-gray-900 dark:bg-[#1C1C1C] dark:text-white">
            <Header />

            <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-10 sm:px-6 lg:px-8">
                <div className="mb-10">
                    <h1 className="mb-2 text-4xl font-bold">All Categories</h1>
                    <p className="text-gray-500 dark:text-gray-400">Browse use cases by category</p>
                </div>

                {loading ? (
                    <div className="flex items-center justify-center py-20">
                        <div className="h-10 w-10 animate-spin rounded-full border-4 border-green-200 border-t-green-600" />
                    </div>
                ) : error ? (
                    <div className="flex min-h-[250px] items-center justify-center rounded-2xl border border-dashed border-red-200 bg-red-50 px-6 py-12 text-center dark:border-red-900 dark:bg-red-950/20">
                        <p className="text-base font-medium text-red-500 dark:text-red-400">
                            Something went wrong. Please try again later.
                        </p>
                    </div>
                ) : categories.length === 0 ? (
                    <div className="flex min-h-[250px] items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-gray-50 px-6 py-12 text-center dark:border-gray-700 dark:bg-[#242424]">
                        <p className="text-base font-medium text-gray-500 dark:text-gray-400">
                            No categories available at the moment.
                        </p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
                        {categories.map((category) => (
                            <Link key={category.id} href={`/categories/${category.slug}`}>
                                <div className="cursor-pointer rounded-2xl border border-gray-100 bg-white p-6 shadow-md transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg dark:border-gray-700 dark:bg-gray-800">
                                    <div className="mb-3 flex items-center">
                                        <span
                                            className="mr-3 inline-block h-4 w-4 flex-shrink-0 rounded-full"
                                            style={{ backgroundColor: category.color }}
                                        />
                                        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                                            {category.name}
                                        </h2>
                                    </div>
                                    {category.description && (
                                        <p className="text-sm text-gray-600 dark:text-gray-300">
                                            {category.description}
                                        </p>
                                    )}
                                </div>
                            </Link>
                        ))}
                    </div>
                )}
            </main>

            <Footer />
        </div>
    );
}
