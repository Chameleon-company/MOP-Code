/**
 * @jest-environment node
 *
 * Tests for /api/blogs/[id]. The PUT cases cover the replaced-cover leak:
 * swapping a cover image used to overwrite cover_img and strand the old
 * object in the bucket with nothing referencing it.
 */

// ==============================
// Mocks — must come before imports
// ==============================

jest.mock("next/server", () => ({
  NextResponse: {
    json: jest.fn().mockImplementation(
      (body: unknown, init?: { status?: number }) => ({
        status: init?.status ?? 200,
        json: jest.fn().mockResolvedValue(body),
        _body: body,
      }),
    ),
  },
}));

jest.mock("@/lib/dbConnect", () => ({
  __esModule: true,
  default: jest.fn(),
}));

jest.mock("@/models/mongoose/Blog", () => ({
  __esModule: true,
  default: {
    findById: jest.fn(),
    findByIdAndDelete: jest.fn(),
  },
}));

jest.mock("@/app/api/library/auth", () => ({
  getAuthUser: jest.fn(),
}));

jest.mock("@/app/api/library/uploadImageToGCS", () => ({
  uploadImageToGCS: jest.fn(),
}));

jest.mock("@/app/api/library/deleteImageFromGCS", () => ({
  deleteImageFromGCS: jest.fn(),
}));

jest.mock("@/app/api/library/gcsBucket", () => ({
  getImagesBucket: jest.fn(() => "test-bucket"),
}));

jest.mock("@/utils/logger", () => ({
  __esModule: true,
  default: { info: jest.fn(), warn: jest.fn(), error: jest.fn() },
}));

// ==============================
// Imports
// ==============================

import { GET, PUT, DELETE } from "@/app/api/blogs/[id]/route";
import Blog from "@/models/mongoose/Blog";
import { getAuthUser } from "@/app/api/library/auth";
import { uploadImageToGCS } from "@/app/api/library/uploadImageToGCS";
import { deleteImageFromGCS } from "@/app/api/library/deleteImageFromGCS";

// ============================================================
// Test helpers
// ============================================================

const BLOG_ID = "507f1f77bcf86cd799439011";
const ADMIN_ID = "507f1f77bcf86cd799439012";
const OLD_COVER = "/api/images/blogs/covers/old.webp";
const NEW_COVER = "/api/images/blogs/covers/new.webp";
const LEGACY_COVER =
  "https://example.supabase.co/storage/v1/object/public/blog-images/blogs/covers/legacy.webp";

const ADMIN_AUTH = {
  userId: ADMIN_ID,
  roleId: 1,
  roleName: "admin",
  isAuthenticated: true,
  isAdmin: true,
};

const USER_AUTH = { ...ADMIN_AUTH, roleId: 2, roleName: "user", isAdmin: false };

const ANON_AUTH = {
  userId: null,
  roleId: null,
  roleName: null,
  isAuthenticated: false,
  isAdmin: false,
};

function makeFile({ type = "image/png", size = 1024 } = {}) {
  return {
    type,
    size,
    arrayBuffer: jest.fn().mockResolvedValue(new ArrayBuffer(8)),
  } as any;
}

function makeRequest(fields: Record<string, unknown> = {}, method = "PUT") {
  const map = new Map(Object.entries(fields));
  return {
    method,
    url: `http://localhost:3000/api/blogs/${BLOG_ID}`,
    headers: { get: (k: string) => (k === "x-user-id" ? ADMIN_ID : null) },
    formData: jest.fn().mockResolvedValue({ get: (k: string) => map.get(k) ?? null }),
  } as any;
}

function makeParams(id: string = BLOG_ID) {
  return { params: Promise.resolve({ id }) };
}

// Mongoose document stand-in: PUT mutates it then calls save().
function makeDoc(coverImg: string = OLD_COVER) {
  const doc: any = {
    _id: { toString: () => BLOG_ID },
    title: "Original title",
    cover_img: coverImg,
    set: jest.fn(),
    save: jest.fn().mockResolvedValue(undefined),
  };
  doc.toObject = () => ({ ...doc });
  return doc;
}

beforeEach(() => {
  jest.clearAllMocks();
  (getAuthUser as jest.Mock).mockReturnValue(ADMIN_AUTH);
  (uploadImageToGCS as jest.Mock).mockResolvedValue(NEW_COVER);
  (deleteImageFromGCS as jest.Mock).mockResolvedValue(true);
});

// ============================================================
// PUT — replaced cover cleanup (regression cover)
// ============================================================

describe("PUT /api/blogs/[id] — replacing the cover", () => {
  test("deletes the previous object after the new URL is saved", async () => {
    const doc = makeDoc(OLD_COVER);
    (Blog.findById as jest.Mock).mockResolvedValue(doc);

    const res: any = await PUT(makeRequest({ cover_img: makeFile() }), makeParams());

    expect(res.status).toBe(200);
    expect(doc.cover_img).toBe(NEW_COVER);
    expect(deleteImageFromGCS).toHaveBeenCalledWith(OLD_COVER, "test-bucket");
    expect(deleteImageFromGCS).toHaveBeenCalledTimes(1);
  });

  test("deletes the old object only after save() succeeds", async () => {
    const order: string[] = [];
    const doc = makeDoc(OLD_COVER);
    doc.save = jest.fn(async () => {
      order.push("save");
    });
    (Blog.findById as jest.Mock).mockResolvedValue(doc);
    (deleteImageFromGCS as jest.Mock).mockImplementation(async () => {
      order.push("delete");
      return true;
    });

    await PUT(makeRequest({ cover_img: makeFile() }), makeParams());

    expect(order).toEqual(["save", "delete"]);
  });

  test("does not delete anything when save() fails", async () => {
    const doc = makeDoc(OLD_COVER);
    doc.save = jest.fn().mockRejectedValue(new Error("save failed"));
    (Blog.findById as jest.Mock).mockResolvedValue(doc);

    const res: any = await PUT(makeRequest({ cover_img: makeFile() }), makeParams());

    expect(res.status).toBe(500);
    expect(deleteImageFromGCS).not.toHaveBeenCalled();
  });

  test("a text-only update leaves the existing cover alone", async () => {
    const doc = makeDoc(OLD_COVER);
    (Blog.findById as jest.Mock).mockResolvedValue(doc);

    const res: any = await PUT(makeRequest({ title: "A new title" }), makeParams());

    expect(res.status).toBe(200);
    expect(doc.cover_img).toBe(OLD_COVER);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
    expect(deleteImageFromGCS).not.toHaveBeenCalled();
  });

  test("a legacy Supabase URL is passed through for the helper to skip", async () => {
    const doc = makeDoc(LEGACY_COVER);
    (Blog.findById as jest.Mock).mockResolvedValue(doc);
    (deleteImageFromGCS as jest.Mock).mockResolvedValue(false);

    const res: any = await PUT(makeRequest({ cover_img: makeFile() }), makeParams());

    expect(res.status).toBe(200);
    expect(deleteImageFromGCS).toHaveBeenCalledWith(LEGACY_COVER, "test-bucket");
  });

  test("a failed cleanup does not fail the request", async () => {
    const doc = makeDoc(OLD_COVER);
    (Blog.findById as jest.Mock).mockResolvedValue(doc);
    (deleteImageFromGCS as jest.Mock).mockRejectedValue(new Error("delete failed"));

    const res: any = await PUT(makeRequest({ cover_img: makeFile() }), makeParams());

    expect(res.status).toBe(200);
  });

  test("upload failure leaves the stored cover untouched", async () => {
    const doc = makeDoc(OLD_COVER);
    (Blog.findById as jest.Mock).mockResolvedValue(doc);
    (uploadImageToGCS as jest.Mock).mockRejectedValue(new Error("GCS unavailable"));

    const res: any = await PUT(makeRequest({ cover_img: makeFile() }), makeParams());

    expect(res.status).toBe(500);
    expect(doc.cover_img).toBe(OLD_COVER);
    expect(doc.save).not.toHaveBeenCalled();
    expect(deleteImageFromGCS).not.toHaveBeenCalled();
  });
});

// ============================================================
// PUT — permissions and validation
// ============================================================

describe("PUT /api/blogs/[id] — permissions and validation", () => {
  test("anonymous request is rejected with 401", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(ANON_AUTH);

    const res: any = await PUT(makeRequest({ cover_img: makeFile() }), makeParams());

    expect(res.status).toBe(401);
    expect(Blog.findById).not.toHaveBeenCalled();
  });

  test("non-admin request is rejected with 403", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(USER_AUTH);

    const res: any = await PUT(makeRequest({ cover_img: makeFile() }), makeParams());

    expect(res.status).toBe(403);
  });

  test("malformed id is rejected with 400", async () => {
    const res: any = await PUT(makeRequest({ cover_img: makeFile() }), makeParams("nope"));

    expect(res.status).toBe(400);
    expect(Blog.findById).not.toHaveBeenCalled();
  });

  test("an empty update is rejected with 400", async () => {
    const res: any = await PUT(makeRequest({}), makeParams());

    expect(res.status).toBe(400);
    expect(res._body.message).toMatch(/at least one field/i);
    expect(Blog.findById).not.toHaveBeenCalled();
  });

  test("unknown id returns 404", async () => {
    (Blog.findById as jest.Mock).mockResolvedValue(null);

    const res: any = await PUT(makeRequest({ title: "A new title" }), makeParams());

    expect(res.status).toBe(404);
  });

  test("oversized replacement cover is rejected before upload", async () => {
    (Blog.findById as jest.Mock).mockResolvedValue(makeDoc());

    const res: any = await PUT(
      makeRequest({ cover_img: makeFile({ size: 5 * 1024 * 1024 + 1 }) }),
      makeParams(),
    );

    expect(res.status).toBe(400);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });
});

// ============================================================
// DELETE
// ============================================================

describe("DELETE /api/blogs/[id]", () => {
  function mockExisting(cover: string | null) {
    (Blog.findById as jest.Mock).mockReturnValue({
      select: jest.fn(() => ({
        lean: jest.fn().mockResolvedValue(cover === null ? null : { cover_img: cover }),
      })),
    });
  }

  test("removes the record and its cover object", async () => {
    mockExisting(OLD_COVER);
    (Blog.findByIdAndDelete as jest.Mock).mockResolvedValue({ _id: BLOG_ID });

    const res: any = await DELETE(makeRequest({}, "DELETE"), makeParams());

    expect(res.status).toBe(200);
    expect(deleteImageFromGCS).toHaveBeenCalledWith(OLD_COVER, "test-bucket");
  });

  test("unknown id returns 404 and removes no object", async () => {
    mockExisting(OLD_COVER);
    (Blog.findByIdAndDelete as jest.Mock).mockResolvedValue(null);

    const res: any = await DELETE(makeRequest({}, "DELETE"), makeParams());

    expect(res.status).toBe(404);
    expect(deleteImageFromGCS).not.toHaveBeenCalled();
  });

  test("non-admin request is rejected with 403", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(USER_AUTH);

    const res: any = await DELETE(makeRequest({}, "DELETE"), makeParams());

    expect(res.status).toBe(403);
    expect(Blog.findByIdAndDelete).not.toHaveBeenCalled();
  });

  test("storage cleanup failure still reports success", async () => {
    mockExisting(OLD_COVER);
    (Blog.findByIdAndDelete as jest.Mock).mockResolvedValue({ _id: BLOG_ID });
    (deleteImageFromGCS as jest.Mock).mockRejectedValue(new Error("delete failed"));

    const res: any = await DELETE(makeRequest({}, "DELETE"), makeParams());

    expect(res.status).toBe(200);
  });
});

// ============================================================
// GET
// ============================================================

describe("GET /api/blogs/[id]", () => {
  test("returns the record", async () => {
    (Blog.findById as jest.Mock).mockReturnValue({
      lean: jest.fn().mockResolvedValue({
        _id: { toString: () => BLOG_ID },
        title: "Original title",
        cover_img: OLD_COVER,
      }),
    });

    const res: any = await GET(makeRequest({}, "GET"), makeParams());

    expect(res.status).toBe(200);
    expect(res._body.data.id).toBe(BLOG_ID);
  });

  test("unknown id returns 404", async () => {
    (Blog.findById as jest.Mock).mockReturnValue({
      lean: jest.fn().mockResolvedValue(null),
    });

    const res: any = await GET(makeRequest({}, "GET"), makeParams());

    expect(res.status).toBe(404);
  });

  test("malformed id returns 400", async () => {
    const res: any = await GET(makeRequest({}, "GET"), makeParams("nope"));

    expect(res.status).toBe(400);
    expect(Blog.findById).not.toHaveBeenCalled();
  });
});
