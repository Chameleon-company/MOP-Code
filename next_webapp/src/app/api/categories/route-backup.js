import { NextResponse } from 'next/server';
import { Pool } from 'pg';
import Category from '../../../../models/Category';

const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
});

const categoryModel = new Category(pool);

export async function GET(request) {
    try {
        const { searchParams } = new URL(request.url);
        const includeInactive = searchParams.get('includeInactive') === 'true';
        const parentId = searchParams.get('parentId');

        const categories = await categoryModel.findAll({
            includeInactive,
            parentId: parentId ? parseInt(parentId) : null
        });

        return NextResponse.json({
            success: true,
            data: categories
        });
    } catch (error) {
        console.error('Error fetching categories:', error);
        return NextResponse.json(
            { success: false, error: 'Failed to fetch categories' },
            { status: 500 }
        );
    }
}

export async function POST(request) {
    try {
        const categoryData = await request.json();
        
        // Validate required fields
        if (!categoryData.name || !categoryData.slug) {
            return NextResponse.json(
                { success: false, error: 'Name and slug are required' },
                { status: 400 }
            );
        }

        const category = await categoryModel.create(categoryData);

        return NextResponse.json({
            success: true,
            data: category
        }, { status: 201 });
    } catch (error) {
        console.error('Error creating category:', error);
        return NextResponse.json(
            { success: false, error: 'Failed to create category' },
            { status: 500 }
        );
    }
}
