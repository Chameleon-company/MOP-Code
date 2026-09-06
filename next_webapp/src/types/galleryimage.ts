// ./types/galleryimage.ts

export interface GalleryImage {
    _id: string;
    legacy_id: string | null;
    title: string | null;
    img_url: string;
    created_by: string | null;
    created_at: string;
    updated_at: string;
}