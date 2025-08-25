import { NextResponse } from 'next/server';
import { Pool } from 'pg';
import Category from '../../../../../models/Category';

const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
});

const categoryModel = new Category(pool);

export async function GET(request, { params }) {
    try {
        const category = await categoryModel.findById(params.id);
        if (!category) {
            return NextResponse.json(
                { success: false, error: 'Category not found' },
                { status: 404 }
            );
        }
        return NextResponse.json({ success: true, data: category });
    } catch (error) {
        console.error('Error fetching category:', error);
        return NextResponse.json(
            { success: false, error: 'Failed to fetch category' },
            { status: 500 }
        );
    }
}

export async function PUT(request, { params }) {
    try {
        const categoryData = await request.json();
        const category = await categoryModel.update(params.id, categoryData);
        return NextResponse.json({ success: true, data: category });
    } catch (error) {
        console.error('Error updating category:', error);
        return NextResponse.json(
            { success: false, error: 'Failed to update category' },
            { status: 500 }
        );
    }
}

export async function DELETE(request, { params }) {
    try {
        await categoryModel.delete(params.id);
        return NextResponse.json({ success: true });
    } catch (error) {
        console.error('Error deleting category:', error);
        return NextResponse.json(
            { success: false, error: 'Failed to delete category' },
            { status: 500 }
        );
    }
}
