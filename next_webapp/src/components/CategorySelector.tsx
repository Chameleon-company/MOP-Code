'use client';

import { useState, useEffect } from 'react';

interface Category {
    id: number;
    name: string;
    slug: string;
    color: string;
    icon?: string;
    description?: string;
}

interface CategorySelectorProps {
    selectedCategories?: string[];
    onCategoryChange: (categories: string[]) => void;
    multiple?: boolean;
}

export default function CategorySelector({ 
    selectedCategories = [], 
    onCategoryChange, 
    multiple = true 
}: CategorySelectorProps) {
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

    const handleCategoryToggle = (categorySlug: string) => {
        if (multiple) {
            const updated = selectedCategories.includes(categorySlug)
                ? selectedCategories.filter(cat => cat !== categorySlug)
                : [...selectedCategories, categorySlug];
            onCategoryChange(updated);
        } else {
            onCategoryChange([categorySlug]);
        }
    };

    if (loading) return <div>Loading categories...</div>;

    return (
        <div className="category-selector">
            <h4 className="font-medium mb-3">Categories</h4>
            <div className="space-y-2">
                {categories.map((category) => (
                    <label key={category.id} className="flex items-center cursor-pointer">
                        <input
                            type={multiple ? "checkbox" : "radio"}
                            name="category"
                            checked={selectedCategories.includes(category.slug)}
                            onChange={() => handleCategoryToggle(category.slug)}
                            className="mr-2"
                        />
                        <span 
                            className="inline-block w-3 h-3 rounded-full mr-2"
                            style={{ backgroundColor: category.color }}
                        ></span>
                        <span className="text-sm">{category.name}</span>
                    </label>
                ))}
            </div>
        </div>
    );
}
