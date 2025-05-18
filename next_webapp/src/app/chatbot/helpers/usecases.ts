export interface LiveUseCase {
  id: number;
  name: string;
  description: string;
  htmlPath: string;   
}

export async function searchUseCases(query: string): Promise<LiveUseCase[]> {
  const res = await fetch(`/api/usecases?q=${encodeURIComponent(query)}`);
  if (!res.ok) return [];
  return (await res.json()) as LiveUseCase[];
}
