/**
 * @jest-environment node
 *
 * Tests for the MongoDB /api/blogs collection routes.
 * GCS and all Mongoose calls are mocked — nothing touches a real bucket or DB.
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
    create: jest.fn(),
    find: jest.fn(),
    countDocuments: jest.fn(),
  },
}));

jest.mock("@/models/mongoose/User", () => ({
  __esModule: true,
  default: { find: jest.fn() },
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

// ==============================
// Imports
// ==============================

import { GET, POST } from "@/app/api/blogs/route";
import Blog from "@/models/mongoose/Blog";
import User from "@/models/mongoose/User";
import { getAuthUser } from "@/app/api/library/auth";
import { uploadImageToGCS } from "@/app/api/library/uploadImageToGCS";
import { deleteImageFromGCS } from "@/app/api/library/deleteImageFromGCS";

// ============================================================
// Test helpers
// ============================================================

const ADMIN_ID = "507f1f77bcf86cd799439011";
const COVER_URL = "/api/images/blogs/covers/new.webp";

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

const VALID_FIELDS = {
  title: "A perfectly good title",
  description: "Short description",
  published_date: "2026-09-18",
  content: "<p>Body copy</p>",
};

function makeFormRequest(
  fields: Record<string, unknown>,
  { userIdHeader = ADMIN_ID }: { userIdHeader?: string | null } = {},
) {
  const map = new Map(Object.entries(fields));
  return {
    method: "POST",
    url: "http://localhost:3000/api/blogs",
    headers: { get: (k: string) => (k === "x-user-id" ? userIdHeader : null) },
    formData: jest.fn().mockResolvedValue({ get: (k: string) => map.get(k) ?? null }),
  } as any;
}

function makeGetRequest(query = "") {
  return {
    method: "GET",
    url: `http://localhost:3000/api/blogs${query}`,
    headers: { get: () => null },
  } as any;
}

function makeCreated(id = "507f1f77bcf86cd799439099") {
  const doc = {
    _id: { toString: () => id },
    title: VALID_FIELDS.title,
    cover_img: COVER_URL,
    created_by: ADMIN_ID,
  };
  return { toObject: () => doc };
}

function mockFindChain(rows: unknown[]) {
  const chain: any = {
    sort: jest.fn(() => chain),
    skip: jest.fn(() => chain),
    limit: jest.fn(() => chain),
    lean: jest.fn().mockResolvedValue(rows),
  };
  (Blog.find as jest.Mock).mockReturnValue(chain);
  (Blog.countDocuments as jest.Mock).mockResolvedValue(rows.length);
  return chain;
}

function mockUserLookup(users: unknown[] = []) {
  (User.find as jest.Mock).mockReturnValue({
    select: jest.fn(() => ({ lean: jest.fn().mockResolvedValue(users) })),
  });
}

beforeEach(() => {
  jest.clearAllMocks();
  (getAuthUser as jest.Mock).mockReturnValue(ADMIN_AUTH);
  (uploadImageToGCS as jest.Mock).mockResolvedValue(COVER_URL);
  (deleteImageFromGCS as jest.Mock).mockResolvedValue(true);
  (Blog.create as jest.Mock).mockResolvedValue(makeCreated());
  mockUserLookup();
});

// ============================================================
// POST /api/blogs — permissions
// ============================================================

describe("POST /api/blogs — permissions", () => {
  test("anonymous request is rejected with 401", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(ANON_AUTH);

    const res: any = await POST(
      makeFormRequest({ ...VALID_FIELDS, cover_img: makeFile() }),
    );

    expect(res.status).toBe(401);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
    expect(Blog.create).not.toHaveBeenCalled();
  });

  test("non-admin request is rejected with 403", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(USER_AUTH);

    const res: any = await POST(
      makeFormRequest({ ...VALID_FIELDS, cover_img: makeFile() }),
    );

    expect(res.status).toBe(403);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });
});

// ============================================================
// POST /api/blogs — validation
// ============================================================

describe("POST /api/blogs — validation", () => {
  test("cover image is required", async () => {
    const res: any = await POST(makeFormRequest({ ...VALID_FIELDS }));

    expect(res.status).toBe(400);
    expect(res._body.errors.cover_img).toBeDefined();
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });

  test("title under 3 characters is rejected", async () => {
    const res: any = await POST(
      makeFormRequest({ ...VALID_FIELDS, title: "ab", cover_img: makeFile() }),
    );

    expect(res.status).toBe(400);
    expect(res._body.errors.title).toMatch(/at least 3/i);
  });

  test("published_date must be ISO formatted", async () => {
    const res: any = await POST(
      makeFormRequest({ ...VALID_FIELDS, published_date: "18-09-2026", cover_img: makeFile() }),
    );

    expect(res.status).toBe(400);
    expect(res._body.errors.published_date).toBeDefined();
  });

  test("content that is only markup counts as empty", async () => {
    const res: any = await POST(
      makeFormRequest({ ...VALID_FIELDS, content: "<p></p>", cover_img: makeFile() }),
    );

    expect(res.status).toBe(400);
    expect(res._body.errors.content).toBeDefined();
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });

  test("disallowed mime type is rejected before upload", async () => {
    const res: any = await POST(
      makeFormRequest({ ...VALID_FIELDS, cover_img: makeFile({ type: "application/pdf" }) }),
    );

    expect(res.status).toBe(400);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });

  test("cover over 5 MB is rejected before upload", async () => {
    const res: any = await POST(
      makeFormRequest({
        ...VALID_FIELDS,
        cover_img: makeFile({ size: 5 * 1024 * 1024 + 1 }),
      }),
    );

    expect(res.status).toBe(400);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });
});

// ============================================================
// POST /api/blogs — upload and persistence
// ============================================================

describe("POST /api/blogs — upload and persistence", () => {
  test("valid admin request stores the cover and returns 201", async () => {
    const res: any = await POST(
      makeFormRequest({ ...VALID_FIELDS, cover_img: makeFile() }),
    );

    expect(res.status).toBe(201);

    const [, filename, bucket] = (uploadImageToGCS as jest.Mock).mock.calls[0];
    expect(filename).toMatch(/^blogs\/covers\//);
    expect(filename).toMatch(/\.webp$/);
    expect(bucket).toBe("test-bucket");

    expect(Blog.create).toHaveBeenCalledWith(
      expect.objectContaining({ title: VALID_FIELDS.title, cover_img: COVER_URL }),
    );
  });

  test("upload failure returns 500 and never writes to the database", async () => {
    (uploadImageToGCS as jest.Mock).mockRejectedValue(new Error("GCS unavailable"));

    const res: any = await POST(
      makeFormRequest({ ...VALID_FIELDS, cover_img: makeFile() }),
    );

    expect(res.status).toBe(500);
    expect(Blog.create).not.toHaveBeenCalled();
  });

  test("database failure deletes the just-uploaded cover so it is not orphaned", async () => {
    (Blog.create as jest.Mock).mockRejectedValue(new Error("write concern failed"));

    const res: any = await POST(
      makeFormRequest({ ...VALID_FIELDS, cover_img: makeFile() }),
    );

    expect(res.status).toBe(500);
    expect(deleteImageFromGCS).toHaveBeenCalledWith(COVER_URL, "test-bucket");
  });

  test("a failed cleanup does not mask the original database error", async () => {
    (Blog.create as jest.Mock).mockRejectedValue(new Error("write concern failed"));
    (deleteImageFromGCS as jest.Mock).mockRejectedValue(new Error("delete failed"));

    const res: any = await POST(
      makeFormRequest({ ...VALID_FIELDS, cover_img: makeFile() }),
    );

    expect(res.status).toBe(500);
  });

  test("created_by falls back to null when the header is not an ObjectId", async () => {
    const res: any = await POST(
      makeFormRequest({ ...VALID_FIELDS, cover_img: makeFile() }, { userIdHeader: "42" }),
    );

    expect(res.status).toBe(201);
    expect(Blog.create).toHaveBeenCalledWith(
      expect.objectContaining({ created_by: null }),
    );
  });
});

// ============================================================
// GET /api/blogs
// ============================================================

describe("GET /api/blogs", () => {
  test("regex metacharacters in search are escaped, not executed", async () => {
    mockFindChain([]);

    await GET(makeGetRequest("?search=.*&search_by=title"));

    const filter = (Blog.find as jest.Mock).mock.calls[0][0];
    expect(filter.title.$regex).toBe("\\.\\*");
  });

  test("default search covers title and description", async () => {
    mockFindChain([]);

    await GET(makeGetRequest("?search=urban"));

    const filter = (Blog.find as jest.Mock).mock.calls[0][0];
    expect(filter.$or).toHaveLength(2);
  });

  test("malformed date_from is rejected", async () => {
    const res: any = await GET(makeGetRequest("?date_from=18-09-2026"));

    expect(res.status).toBe(400);
    expect(Blog.find).not.toHaveBeenCalled();
  });

  test("non-ObjectId created_by filter is rejected", async () => {
    const res: any = await GET(makeGetRequest("?created_by=42"));

    expect(res.status).toBe(400);
    expect(Blog.find).not.toHaveBeenCalled();
  });

  test("resolves the author name for each row", async () => {
    mockFindChain([
      {
        _id: { toString: () => "b1" },
        title: "One",
        created_by: ADMIN_ID,
      },
    ]);
    mockUserLookup([
      {
        _id: { toString: () => ADMIN_ID },
        profile: { first_name: "Ishika", last_name: "Mandal" },
      },
    ]);

    const res: any = await GET(makeGetRequest());

    expect(res.status).toBe(200);
    expect(res._body.data[0].created_by_name).toBe("Ishika Mandal");
  });

  test("rows without an author fall back to Admin", async () => {
    mockFindChain([{ _id: { toString: () => "b1" }, title: "One", created_by: null }]);

    const res: any = await GET(makeGetRequest());

    expect(res._body.data[0].created_by_name).toBe("Admin");
    expect(User.find).not.toHaveBeenCalled();
  });

  test("returns pagination metadata", async () => {
    mockFindChain([{ _id: { toString: () => "b1" }, title: "One", created_by: null }]);

    const res: any = await GET(makeGetRequest("?page=1&pageSize=10"));

    expect(res._body.pagination).toEqual(
      expect.objectContaining({ page: 1, pageSize: 10, total: 1 }),
    );
  });
});
