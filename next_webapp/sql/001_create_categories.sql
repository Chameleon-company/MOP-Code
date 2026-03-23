-- Create categories table
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; -- To generate UUIDs

CREATE TABLE categories (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    slug VARCHAR(255) NOT NULL UNIQUE,
    color VARCHAR(7), -- For UI theming (hex color)
    icon VARCHAR(100), -- Icon name/class
    parent_id BIGINT REFERENCES categories(id) ON DELETE SET NULL, -- For hierarchical categories
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_categories_slug ON categories(slug);
CREATE INDEX idx_categories_parent_id ON categories(parent_id);
CREATE INDEX idx_categories_active ON categories(is_active);
