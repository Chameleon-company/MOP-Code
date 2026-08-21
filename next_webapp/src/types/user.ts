// ./types/user.ts

export interface RoleRef {
    id?: string;
    legacy_id: string | null;
    role_name: string | null;
}

export interface Profile {
    id?: string;
    first_name: string | null;
    last_name: string | null;
    age: number | null;
    gender: string
    profile_img: string | null;
    updated_at: Date | null;
}

export interface User {
    _id: string;
    legacy_id: string | null;
    email: string;
    password: string;
    role: RoleRef | null;
    profile: Profile | null;
    created_at: Date;
    updated_at: Date;
}