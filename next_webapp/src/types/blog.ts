// ./types/blog.ts

export interface Blog {
    _id: string;
    legacy_id: string | null;
    title: string;
    content: string | null;
    description: string | null;
    cover_img: string | null;
    published_date: string | null;
    created_by: string | null;
    created_at: string;
    updated_at: string;
}