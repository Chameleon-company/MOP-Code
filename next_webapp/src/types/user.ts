// ./types/user.ts

export interface RoleRef {
    legacy_id: string | null;
    role_name: string | null;
}

export interface Profile {
    first_name: string | null;
    last_name: string | null;
    age: number | null;
    gender: string
    profile_img: string | null;
    updated_at: string | null;
}

export interface User {
    _id: string;
    legacy_id: string | null;
    email: string;
    password: string;
    role: RoleRef | null;
    profile: Profile | null;
    created_at: string;
    updated_at: string;
}