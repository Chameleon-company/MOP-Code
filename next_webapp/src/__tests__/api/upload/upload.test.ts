/**
 * @jest-environment node
 *
 * Tests for the shared /api/upload route used by the Category and Use Case
 * admin pages. Covers auth, validation, the folder allowlist and upload failure.
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

jest.mock("@/app/api/library/auth", () => ({
  getAuthUser: jest.fn(),
}));

jest.mock("@/app/api/library/uploadImageToGCS", () => ({
  uploadImageToGCS: jest.fn(),
}));

jest.mock("@/app/api/library/gcsBucket", () => ({
  getImagesBucket: jest.fn(() => "test-bucket"),
}));

// ==============================
// Imports
// ==============================

import { POST } from "@/app/api/upload/route";
import { getAuthUser } from "@/app/api/library/auth";
import { uploadImageToGCS } from "@/app/api/library/uploadImageToGCS";

// ============================================================
// Test helpers
// ============================================================

const USER_ID = "507f1f77bcf86cd799439011";

const AUTHED = {
  userId: USER_ID,
  roleId: 1,
  roleName: "admin",
  isAuthenticated: true,
  isAdmin: true,
};

const ANON = {
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
    name: "photo.png",
    arrayBuffer: jest.fn().mockResolvedValue(new ArrayBuffer(8)),
  } as any;
}

function makeRequest(fields: Record<string, unknown>) {
  const map = new Map(Object.entries(fields));
  return {
    method: "POST",
    url: "http://localhost:3000/api/upload",
    headers: { get: (k: string) => (k === "x-user-id" ? USER_ID : null) },
    formData: jest.fn().mockResolvedValue({ get: (k: string) => map.get(k) ?? null }),
  } as any;
}

beforeEach(() => {
  // clearAllMocks resets calls but not implementations, so restore the bucket
  // stub explicitly — one test replaces it with a throwing version.
  jest.clearAllMocks();
  (getAuthUser as jest.Mock).mockReturnValue(AUTHED);
  (uploadImageToGCS as jest.Mock).mockResolvedValue("/api/images/categories/x.webp");
  jest
    .requireMock("@/app/api/library/gcsBucket")
    .getImagesBucket.mockImplementation(() => "test-bucket");
});

// ============================================================
// Permissions
// ============================================================

describe("POST /api/upload — permissions", () => {
  test("anonymous request is rejected with 401", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(ANON);

    const res: any = await POST(makeRequest({ file: makeFile(), folder: "categories" }));

    expect(res.status).toBe(401);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });
});

// ============================================================
// Validation
// ============================================================

describe("POST /api/upload — validation", () => {
  test("missing file returns 400", async () => {
    const res: any = await POST(makeRequest({ folder: "categories" }));

    expect(res.status).toBe(400);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });

  test("disallowed mime type returns 400", async () => {
    const res: any = await POST(
      makeRequest({ file: makeFile({ type: "text/html" }), folder: "categories" }),
    );

    expect(res.status).toBe(400);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });

  test("file over 5 MB returns 400", async () => {
    const res: any = await POST(
      makeRequest({ file: makeFile({ size: 5 * 1024 * 1024 + 1 }), folder: "categories" }),
    );

    expect(res.status).toBe(400);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });

  test("unknown bucket value is rejected", async () => {
    const res: any = await POST(
      makeRequest({ file: makeFile(), folder: "categories", bucket: "something-else" }),
    );

    expect(res.status).toBe(400);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });
});

// ============================================================
// Folder handling — the only caller-controlled part of the object path
// ============================================================

describe("POST /api/upload — folder handling", () => {
  test.each([
    ["../../etc/passwd"],
    ["nested/path"],
    ["has space"],
    [".."],
  ])("rejects folder %p", async (folder) => {
    const res: any = await POST(makeRequest({ file: makeFile(), folder }));

    expect(res.status).toBe(400);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });

  test("accepts a plain folder segment and builds the object path from it", async () => {
    const res: any = await POST(makeRequest({ file: makeFile(), folder: "categories" }));

    expect(res.status).toBe(200);

    const [, filename, bucket] = (uploadImageToGCS as jest.Mock).mock.calls[0];
    expect(filename).toMatch(/^categories\//);
    expect(filename).toMatch(/\.webp$/);
    expect(filename).toContain(USER_ID);
    expect(bucket).toBe("test-bucket");
  });

  test("stored extension is always .webp regardless of the uploaded type", async () => {
    await POST(makeRequest({ file: makeFile({ type: "image/jpeg" }), folder: "categories" }));

    const [, filename] = (uploadImageToGCS as jest.Mock).mock.calls[0];
    expect(filename.endsWith(".webp")).toBe(true);
  });
});

// ============================================================
// Failure handling
// ============================================================

describe("POST /api/upload — failures", () => {
  test("upload failure returns 500", async () => {
    (uploadImageToGCS as jest.Mock).mockRejectedValue(new Error("GCS unavailable"));

    const res: any = await POST(makeRequest({ file: makeFile(), folder: "categories" }));

    expect(res.status).toBe(500);
  });

  test("an unset bucket env var surfaces as a 500 rather than a default bucket", async () => {
    const { getImagesBucket } = jest.requireMock("@/app/api/library/gcsBucket");
    getImagesBucket.mockImplementationOnce(() => {
      throw new Error("GCS_IMAGES_BUCKET is not set");
    });

    const res: any = await POST(makeRequest({ file: makeFile(), folder: "categories" }));

    expect(res.status).toBe(500);
  });

  test("successful upload returns the proxy URL", async () => {
    const res: any = await POST(makeRequest({ file: makeFile(), folder: "categories" }));

    expect(res.status).toBe(200);
    expect(res._body).toEqual({ success: true, url: "/api/images/categories/x.webp" });
  });
});
