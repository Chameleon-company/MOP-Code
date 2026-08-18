// Map a Mongo document (or .lean() object) to the flat shape the frontend
// expects — plain string `id`, never a raw `_id`/`__v`. content_file_id and
// content_type are metadata only; the actual notebook/HTML bytes are never
// included here — fetch those from GET /api/usecases/[id]/content.
export function toUseCaseDTO(doc: any) {
  const { _id, __v, ...rest } = doc;
  return { id: _id.toString(), ...rest };
}
