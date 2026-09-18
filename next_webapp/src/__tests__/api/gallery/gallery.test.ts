/**
 * @jest-environment node
 *
 * Tests for the MongoDB /api/gallery collection routes.
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

jest.mock("@/models/mongoose/GalleryImage", () => ({
  __esModule: true,
  default: {
    create: jest.fn(),
    find: jest.fn(),
    countDocuments: jest.fn(),
  },
}));

jest.mock("@/app/api/library/auth", () => ({
  getAuthUser: jest.fn(),
}));

jest.mock("@/app/api/library/errorResponse", () => ({
  errorResponse: jest.fn().mockImplementation(
    (message: string, status: number, code: string) => ({
      status,
      json: jest.fn().mockResolvedValue({ success: false, message, code }),
      _body: { success: false, message, code },
    }),
  ),
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

import { GET, POST } from "@/app/api/gallery/route";
import GalleryImage from "@/models/mongoose/GalleryImage";
import { getAuthUser } from "@/app/api/library/auth";
import { uploadImageToGCS } from "@/app/api/library/uploadImageToGCS";
import { deleteImageFromGCS } from "@/app/api/library/deleteImageFromGCS";

// ============================================================
// Test helpers
// ============================================================

const ADMIN_ID = "507f1f77bcf86cd799439011";

const ADMIN_AUTH = {
  userId: ADMIN_ID,
  roleId: 1,
  roleName: "admin",
  isAuthenticated: true,
  isAdmin: true,
};

const USER_AUTH = {
  userId: "507f1f77bcf86cd799439012",
  roleId: 2,
  roleName: "user",
  isAuthenticated: true,
  isAdmin: false,
};

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

// The routes read multipart fields via request.formData().get(name) and the
// user id header via request.headers.get("x-user-id").
function makeFormRequest(
  fields: Record<string, unknown>,
  { userIdHeader = ADMIN_ID }: { userIdHeader?: string | null } = {},
) {
  const map = new Map(Object.entries(fields));
  return {
    method: "POST",
    url: "http://localhost:3000/api/gallery",
    headers: { get: (k: string) => (k === "x-user-id" ? userIdHeader : null) },
    formData: jest.fn().mockResolvedValue({ get: (k: string) => map.get(k) ?? null }),
  } as any;
}

function makeGetRequest(query = "") {
  return {
    method: "GET",
    url: `http://localhost:3000/api/gallery${query}`,
    headers: { get: () => null },
  } as any;
}

function makeCreated(id = "507f1f77bcf86cd799439099", overrides = {}) {
  const doc = {
    _id: { toString: () => id },
    title: "A picture",
    img_url: "/api/images/gallery/new.webp",
    created_by: ADMIN_ID,
    ...overrides,
  };
  return { toObject: () => doc };
}

beforeEach(() => {
  jest.clearAllMocks();
  (getAuthUser as jest.Mock).mockReturnValue(ADMIN_AUTH);
  (uploadImageToGCS as jest.Mock).mockResolvedValue("/api/images/gallery/new.webp");
  (deleteImageFromGCS as jest.Mock).mockResolvedValue(true);
  (GalleryImage.create as jest.Mock).mockResolvedValue(makeCreated());
});

// ============================================================
// POST /api/gallery — permission checks
// ============================================================

describe("POST /api/gallery — permissions", () => {
  test("anonymous request is rejected with 401", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(ANON_AUTH);

    const res: any = await POST(makeFormRequest({ title: "x", image: makeFile() }));

    expect(res.status).toBe(401);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
    expect(GalleryImage.create).not.toHaveBeenCalled();
  });

  test("non-admin request is rejected with 403", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(USER_AUTH);

    const res: any = await POST(makeFormRequest({ title: "x", image: makeFile() }));

    expect(res.status).toBe(403);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });
});

// ============================================================
// POST /api/gallery — validation
// ============================================================

describe("POST /api/gallery — validation", () => {
  test("missing title returns 400 and never uploads", async () => {
    const res: any = await POST(makeFormRequest({ image: makeFile() }));

    expect(res.status).toBe(400);
    expect(res._body.errors.title).toBeDefined();
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });

  test("missing image returns 400", async () => {
    const res: any = await POST(makeFormRequest({ title: "A picture" }));

    expect(res.status).toBe(400);
    expect(res._body.errors.image).toBeDefined();
  });

  test("disallowed mime type is rejected before upload", async () => {
    const res: any = await POST(
      makeFormRequest({ title: "A picture", image: makeFile({ type: "application/pdf" }) }),
    );

    expect(res.status).toBe(400);
    expect(res._body.errors.image).toMatch(/JPEG, PNG, or WebP/i);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });

  test("file over 5 MB is rejected before upload", async () => {
    const res: any = await POST(
      makeFormRequest({
        title: "A picture",
        image: makeFile({ size: 5 * 1024 * 1024 + 1 }),
      }),
    );

    expect(res.status).toBe(400);
    expect(res._body.errors.image).toMatch(/under 5 MB/i);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });
});

// ============================================================
// POST /api/gallery — happy path and failure modes
// ============================================================

describe("POST /api/gallery — upload and persistence", () => {
  test("valid admin request stores the image and returns 201", async () => {
    const res: any = await POST(makeFormRequest({ title: "A picture", image: makeFile() }));

    expect(res.status).toBe(201);
    expect(uploadImageToGCS).toHaveBeenCalledTimes(1);

    // filename must land under the gallery/ prefix the proxy allowlists
    const [, filename, bucket] = (uploadImageToGCS as jest.Mock).mock.calls[0];
    expect(filename).toMatch(/^gallery\//);
    expect(filename).toMatch(/\.webp$/);
    expect(bucket).toBe("test-bucket");

    expect(GalleryImage.create).toHaveBeenCalledWith(
      expect.objectContaining({ title: "A picture", img_url: "/api/images/gallery/new.webp" }),
    );
    expect(res._body.data.id).toBe("507f1f77bcf86cd799439099");
  });

  test("upload failure returns 500 and never writes to the database", async () => {
    (uploadImageToGCS as jest.Mock).mockRejectedValue(new Error("GCS unavailable"));

    const res: any = await POST(makeFormRequest({ title: "A picture", image: makeFile() }));

    expect(res.status).toBe(500);
    expect(GalleryImage.create).not.toHaveBeenCalled();
  });

  test("database failure deletes the just-uploaded image so it is not orphaned", async () => {
    (GalleryImage.create as jest.Mock).mockRejectedValue(new Error("write concern failed"));

    const res: any = await POST(makeFormRequest({ title: "A picture", image: makeFile() }));

    expect(res.status).toBe(500);
    expect(deleteImageFromGCS).toHaveBeenCalledWith(
      "/api/images/gallery/new.webp",
      "test-bucket",
    );
  });

  test("a failed cleanup does not mask the original database error", async () => {
    (GalleryImage.create as jest.Mock).mockRejectedValue(new Error("write concern failed"));
    (deleteImageFromGCS as jest.Mock).mockRejectedValue(new Error("delete failed"));

    const res: any = await POST(makeFormRequest({ title: "A picture", image: makeFile() }));

    expect(res.status).toBe(500);
  });

  test("created_by falls back to null when the header is not an ObjectId", async () => {
    const res: any = await POST(
      makeFormRequest({ title: "A picture", image: makeFile() }, { userIdHeader: "42" }),
    );

    expect(res.status).toBe(201);
    expect(GalleryImage.create).toHaveBeenCalledWith(
      expect.objectContaining({ created_by: null }),
    );
  });
});

// ============================================================
// GET /api/gallery
// ============================================================

describe("GET /api/gallery", () => {
  function mockFindChain(rows: unknown[]) {
    const chain: any = {
      select: jest.fn(() => chain),
      sort: jest.fn(() => chain),
      skip: jest.fn(() => chain),
      limit: jest.fn(() => chain),
      lean: jest.fn().mockResolvedValue(rows),
    };
    (GalleryImage.find as jest.Mock).mockReturnValue(chain);
    (GalleryImage.countDocuments as jest.Mock).mockResolvedValue(rows.length);
    return chain;
  }

  test("anonymous request is rejected with 401", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(ANON_AUTH);

    const res: any = await GET(makeGetRequest());

    expect(res.status).toBe(401);
    expect(GalleryImage.find).not.toHaveBeenCalled();
  });

  test("regex metacharacters in search are escaped, not executed", async () => {
    mockFindChain([]);

    await GET(makeGetRequest("?search=.*"));

    const filter = (GalleryImage.find as jest.Mock).mock.calls[0][0];
    expect(filter.title.$regex).toBe("\\.\\*");
  });

  test("pageSize above the maximum is rejected", async () => {
    const res: any = await GET(makeGetRequest("?pageSize=1000"));

    expect(res.status).toBe(400);
    expect(GalleryImage.find).not.toHaveBeenCalled();
  });

  test("returns rows with pagination metadata", async () => {
    mockFindChain([
      { _id: { toString: () => "a" }, title: "one", img_url: "/api/images/gallery/1.webp" },
    ]);

    const res: any = await GET(makeGetRequest("?page=1&pageSize=12"));

    expect(res.status).toBe(200);
    expect(res._body.data).toHaveLength(1);
    expect(res._body.pagination).toEqual(
      expect.objectContaining({ page: 1, pageSize: 12, total: 1 }),
    );
  });
});
