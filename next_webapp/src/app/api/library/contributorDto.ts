// Map a Mongo document (or .lean() object) to the flat shape the frontend
// expects — plain string `id`, never a raw `_id`/`__v`.
export function toContributorDTO(doc: any) {
  const { _id, __v, ...rest } = doc;
  return { id: _id.toString(), ...rest };
}
