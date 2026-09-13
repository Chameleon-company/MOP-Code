// ./types/log.ts

export interface Log {
    _id: string;
    legacy_id: string | null;
    level: string;
    message: string;
    timestamp: string;
    meta: Record<string, unknown> | null;
    source: string | null;
    user_id: string | null;
    ip_address: string | null;
    user_agent: string | null;
    method: string | null;
    url: string | null;
    status_code: number | null;
    response_time: number | null;
    created_at: string;
}