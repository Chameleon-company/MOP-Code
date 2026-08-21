// ./types/role.ts

export interface Role {
    _id: string;
    legacy_id: string | null;
    role_name: string;
    created_at: Date;
    updated_at: Date;
}