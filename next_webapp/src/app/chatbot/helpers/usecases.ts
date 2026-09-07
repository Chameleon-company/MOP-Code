import { apiFetch } from "@/lib/apiFetch";

export interface LiveUseCase {
  id: number;
  name: string;
  description: string;
  htmlPath: string;
}

export async function searchUseCases(query: string): Promise<LiveUseCase[]> {
  try {
    // silent: true - this is a lookup helper; callers expect a plain
    // array back and fall back to [] on failure, not a toast.
    return await apiFetch<LiveUseCase[]>(`/api/usecases?q=${encodeURIComponent(query)}`, { silent: true });
  } catch {
    return [];
  }
}


