// ./types/usecase.ts

export interface CategoryRef {
    id?: string;
    legacy_id: string | null;
    category_name: string | null;
}

export interface TagRef {
    id?: string;
    legacy_id: string | null;
    name: string | null;
    slug: string | null;
}

export interface UseCase {
    _id: string;
    legacy_id: string | null;
    title: string;
    description: string | null;
    cover_img: string | null;
    content_file_id: string | null;
    content_type: "notebook" | "html" | null;
    category: CategoryRef | null;
    tags: TagRef[];
    created_by: string | null;
    created_at: Date;
    updated_at: Date;
}