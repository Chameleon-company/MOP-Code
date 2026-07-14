# Search Feature Implementation Guide

This guide documents the implementation of the Category Table & Search Filter Integration feature.

## Overview

The search feature includes:
- Category management with PostgreSQL database
- Advanced search with category filtering
- React components for search and filtering
- RESTful API endpoints

## Database Setup

### 1. Run SQL Migration Files

Execute the SQL files in order:

```bash
# Connect to your PostgreSQL database and run:
psql -U your_username -d your_database -f sql/001_create_categories.sql
psql -U your_username -d your_database -f sql/002_add_category_to_posts.sql
psql -U your_username -d your_database -f sql/003_seed_categories.sql
```

### 2. Environment Configuration

Update your `.env` file with PostgreSQL credentials:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/mop_database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mop_database
POSTGRES_USER=username
POSTGRES_PASSWORD=password
```

## API Endpoints

### Categories API
- `GET /api/categories` - Fetch all categories
- `POST /api/categories` - Create new category
- `GET /api/categories/[id]` - Get category by ID
- `PUT /api/categories/[id]` - Update category
- `DELETE /api/categories/[id]` - Delete category

### Search API
- `GET /api/search` - Search with filters

Query parameters:
- `q` - Search query
- `category` - Single category filter
- `categories` - Multiple categories filter
- `page` - Page number
- `limit` - Results per page
- `sortBy` - Sort field (created_at, title, view_count)
- `sortOrder` - Sort direction (ASC, DESC)

## Frontend Components

### SearchFilter Component
Location: `src/components/SearchFilter.tsx`
- Provides search input and category filtering
- Updates URL parameters
- Handles sort options

### SearchResults Component
Location: `src/components/SearchResults.tsx`
- Displays search results
- Shows category badges
- Handles loading states

### CategorySelector Component
Location: `src/components/CategorySelector.tsx`
- Reusable category selection component
- Supports single or multiple selection

## Pages

### Search Page
Location: `src/app/[locale]/search/page.tsx`
- Main search interface
- Combines SearchFilter and SearchResults
- Handles pagination

### Categories Page
Location: `src/app/[locale]/categories/page.tsx`
- Lists all categories
- Links to individual category pages

### Individual Category Page
Location: `src/app/[locale]/categories/[slug]/page.tsx`
- Shows posts for a specific category
- Dynamic routing by category slug

## File Structure

```
next_webapp/
├── sql/
│   ├── 001_create_categories.sql
│   ├── 002_add_category_to_posts.sql
│   └── 003_seed_categories.sql
├── models/
│   └── Category.js
├── lib/
│   └── postgresql.ts
├── src/
│   ├── app/
│   │   ├── api/
│   │   │   ├── categories/
│   │   │   │   ├── route.js
│   │   │   │   └── [id]/route.js
│   │   │   └── search/
│   │   │       └── route.js
│   │   └── [locale]/
│   │       ├── search/
│   │       │   └── page.tsx
│   │       └── categories/
│   │           ├── page.tsx
│   │           └── [slug]/page.tsx
│   └── components/
│       ├── SearchFilter.tsx
│       ├── SearchResults.tsx
│       └── CategorySelector.tsx
└── .env
```

## Usage Examples

### Basic Search
```
/search?q=technology
```

### Search with Category Filter
```
/search?q=data&category=technology
```

### Multiple Categories
```
/search?categories=technology&categories=environment
```

### Sorted Results
```
/search?q=machine+learning&sortBy=created_at&sortOrder=DESC
```

## Development Notes

1. **Database Migration**: Run SQL files in the correct order
2. **Environment Variables**: Update `.env` with your PostgreSQL credentials
3. **Dependencies**: Install `pg` and `@types/pg` packages
4. **TypeScript**: All components include proper type definitions
5. **Error Handling**: API endpoints include comprehensive error handling

## Next Steps

1. Set up PostgreSQL database
2. Run migration scripts
3. Update environment variables
4. Test API endpoints
5. Verify frontend components work correctly

## Troubleshooting

### Common Issues

1. **Database Connection**: Verify DATABASE_URL in .env file
2. **Missing Tables**: Ensure SQL migration files are executed
3. **TypeScript Errors**: Check component prop types
4. **API Errors**: Verify PostgreSQL is running and accessible

### Testing the Implementation

1. Start the development server: `npm run dev`
2. Navigate to `/search` to test the search functionality
3. Navigate to `/categories` to view all categories
4. Test category filtering and search combinations
