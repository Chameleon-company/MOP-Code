import { NextResponse } from 'next/server';
import { Pool } from 'pg';

const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
});

export async function GET(request) {
    try {
        const { searchParams } = new URL(request.url);
        const query = searchParams.get('q') || '';
        const category = searchParams.get('category');
        const categories = searchParams.getAll('categories'); // For multiple categories
        const page = parseInt(searchParams.get('page')) || 1;
        const limit = parseInt(searchParams.get('limit')) || 10;
        const sortBy = searchParams.get('sortBy') || 'created_at';
        const sortOrder = searchParams.get('sortOrder') || 'DESC';

        const offset = (page - 1) * limit;

        // Build dynamic query
        let searchQuery = `
            SELECT 
                p.*,
                c.name as category_name,
                c.slug as category_slug,
                c.color as category_color,
                c.icon as category_icon
            FROM posts p
            LEFT JOIN categories c ON p.category_id = c.id
            WHERE 1=1
        `;
        
        const queryParams = [];
        let paramCount = 0;

        // Add text search filter
        if (query.trim()) {
            paramCount++;
            searchQuery += ` AND (
                p.title ILIKE $${paramCount} OR 
                p.content ILIKE $${paramCount} OR 
                p.description ILIKE $${paramCount}
            )`;
            queryParams.push(`%${query}%`);
        }

        // Add single category filter
        if (category) {
            paramCount++;
            searchQuery += ` AND c.slug = $${paramCount}`;
            queryParams.push(category);
        }

        // Add multiple categories filter
        if (categories.length > 0) {
            const placeholders = categories.map(() => {
                paramCount++;
                return `$${paramCount}`;
            }).join(',');
            searchQuery += ` AND c.slug IN (${placeholders})`;
            queryParams.push(...categories);
        }

        // Add active filter
        searchQuery += ` AND p.is_active = true AND (c.is_active = true OR c.is_active IS NULL)`;

        // Add sorting
        const validSortFields = ['created_at', 'updated_at', 'title', 'view_count'];
        const validSortOrders = ['ASC', 'DESC'];
        
        if (validSortFields.includes(sortBy) && validSortOrders.includes(sortOrder.toUpperCase())) {
            searchQuery += ` ORDER BY p.${sortBy} ${sortOrder.toUpperCase()}`;
        } else {
            searchQuery += ` ORDER BY p.created_at DESC`;
        }

        // Add pagination
        paramCount++;
        searchQuery += ` LIMIT $${paramCount}`;
        queryParams.push(limit);

        paramCount++;
        searchQuery += ` OFFSET $${paramCount}`;
        queryParams.push(offset);

        // Execute search query
        const searchResult = await pool.query(searchQuery, queryParams);

        // Get total count for pagination
        let countQuery = `
            SELECT COUNT(*) as total
            FROM posts p
            LEFT JOIN categories c ON p.category_id = c.id
            WHERE 1=1
        `;
        
        const countParams = [];
        let countParamIndex = 0;

        // Apply same filters for count query
        if (query.trim()) {
            countParamIndex++;
            countQuery += ` AND (
                p.title ILIKE $${countParamIndex} OR 
                p.content ILIKE $${countParamIndex} OR 
                p.description ILIKE $${countParamIndex}
            )`;
            countParams.push(`%${query}%`);
        }

        if (category) {
            countParamIndex++;
            countQuery += ` AND c.slug = $${countParamIndex}`;
            countParams.push(category);
        }

        if (categories.length > 0) {
            const placeholders = categories.map(() => {
                countParamIndex++;
                return `$${countParamIndex}`;
            }).join(',');
            countQuery += ` AND c.slug IN (${placeholders})`;
            countParams.push(...categories);
        }

        countQuery += ` AND p.is_active = true AND (c.is_active = true OR c.is_active IS NULL)`;

        const countResult = await pool.query(countQuery, countParams);
        const total = parseInt(countResult.rows[0].total);

        return NextResponse.json({
            success: true,
            data: {
                results: searchResult.rows,
                pagination: {
                    page,
                    limit,
                    total,
                    totalPages: Math.ceil(total / limit),
                    hasNext: page < Math.ceil(total / limit),
                    hasPrev: page > 1
                },
                filters: {
                    query,
                    category,
                    categories,
                    sortBy,
                    sortOrder
                }
            }
        });

    } catch (error) {
        console.error('Search API error:', error);
        return NextResponse.json(
            { success: false, error: 'Search failed' },
            { status: 500 }
        );
    }
}
