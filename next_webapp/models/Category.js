const { Pool } = require('pg');

class Category {
    constructor(pool) {
        this.pool = pool;
    }

    async findAll(options = {}) {
        const { includeInactive = false, parentId = null } = options;
        
        let query = 'SELECT * FROM categories WHERE 1=1';
        const params = [];
        let paramCount = 0;

        if (!includeInactive) {
            query += ' AND is_active = true';
        }

        if (parentId !== null) {
            paramCount++;
            query += ` AND parent_id = $${paramCount}`;
            params.push(parentId);
        }

        query += ' ORDER BY name ASC';

        const result = await this.pool.query(query, params);
        return result.rows;
    }

    async findById(id) {
        const result = await this.pool.query(
            'SELECT * FROM categories WHERE id = $1',
            [id]
        );
        return result.rows[0];
    }

    async findBySlug(slug) {
        const result = await this.pool.query(
            'SELECT * FROM categories WHERE slug = $1 AND is_active = true',
            [slug]
        );
        return result.rows[0];
    }

    async create(categoryData) {
        const { name, description, slug, color, icon, parentId } = categoryData;
        
        const result = await this.pool.query(
            `INSERT INTO categories (name, description, slug, color, icon, parent_id)
             VALUES ($1, $2, $3, $4, $5, $6)
             RETURNING *`,
            [name, description, slug, color, icon, parentId]
        );
        return result.rows[0];
    }

    async update(id, categoryData) {
        const { name, description, slug, color, icon, parentId, isActive } = categoryData;
        
        const result = await this.pool.query(
            `UPDATE categories 
             SET name = $1, description = $2, slug = $3, color = $4, 
                 icon = $5, parent_id = $6, is_active = $7, updated_at = NOW()
             WHERE id = $8
             RETURNING *`,
            [name, description, slug, color, icon, parentId, isActive, id]
        );
        return result.rows[0];
    }

    async delete(id) {
        const result = await this.pool.query(
            'DELETE FROM categories WHERE id = $1 RETURNING *',
            [id]
        );
        return result.rows[0];
    }
}

module.exports = Category;
