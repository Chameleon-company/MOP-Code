/**
 * @jest-environment node
 *
 * Tests for /api/gallery/[id]. The PUT cases cover the replaced-image leak:
 * swapping an image used to overwrite img_url and strand the old object in
 * the bucket with nothing referencing it.
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
    findById: jest.fn(),
    findByIdAndDelete: jest.fn(),
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

jest.mock("@/utils/logger", () => ({
  __esModule: true,
  default: { info: jest.fn(), warn: jest.fn(), error: jest.fn() },
}));

// ==============================
// Imports
// ==============================

import { GET, PUT, DELETE } from "@/app/api/gallery/[id]/route";
import GalleryImage from "@/models/mongoose/GalleryImage";
import { getAuthUser } from "@/app/api/library/auth";
import { uploadImageToGCS } from "@/app/api/library/uploadImageToGCS";
import { deleteImageFromGCS } from "@/app/api/library/deleteImageFromGCS";

// ============================================================
// Test helpers
// ============================================================

const IMAGE_ID = "507f1f77bcf86cd799439011";
const ADMIN_ID = "507f1f77bcf86cd799439012";
const OLD_URL = "/api/images/gallery/old.webp";
const NEW_URL = "/api/images/gallery/new.webp";
const LEGACY_URL =
  "https://example.supabase.co/storage/v1/object/public/gallery-images/gallery/legacy.webp";

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
    url: `http://localhost:3000/api/gallery/${IMAGE_ID}`,
    headers: { get: (k: string) => (k === "x-user-id" ? ADMIN_ID : null) },
    formData: jest.fn().mockResolvedValue({ get: (k: string) => map.get(k) ?? null }),
  } as any;
}

function makeParams(id: string = IMAGE_ID) {
  return { params: Promise.resolve({ id }) };
}

// Mongoose document stand-in: PUT mutates it then calls save().
function makeDoc(imgUrl: string = OLD_URL) {
  const doc: any = {
    _id: { toString: () => IMAGE_ID },
    title: "Original title",
    img_url: imgUrl,
    save: jest.fn().mockResolvedValue(undefined),
  };
  doc.toObject = () => ({ ...doc });
  return doc;
}

beforeEach(() => {
  jest.clearAllMocks();
  (getAuthUser as jest.Mock).mockReturnValue(ADMIN_AUTH);
  (uploadImageToGCS as jest.Mock).mockResolvedValue(NEW_URL);
  (deleteImageFromGCS as jest.Mock).mockResolvedValue(true);
});

// ============================================================
// PUT — replaced image cleanup (regression cover)
// ============================================================

describe("PUT /api/gallery/[id] — replacing the image", () => {
  test("deletes the previous object after the new URL is saved", async () => {
    const doc = makeDoc(OLD_URL);
    (GalleryImage.findById as jest.Mock).mockResolvedValue(doc);

    const res: any = await PUT(makeRequest({ image: makeFile() }), makeParams());

    expect(res.status).toBe(200);
    expect(doc.img_url).toBe(NEW_URL);
    expect(deleteImageFromGCS).toHaveBeenCalledWith(OLD_URL, "test-bucket");
    expect(deleteImageFromGCS).toHaveBeenCalledTimes(1);
  });

  test("deletes the old object only after save() succeeds", async () => {
    const order: string[] = [];
    const doc = makeDoc(OLD_URL);
    doc.save = jest.fn(async () => {
      order.push("save");
    });
    (GalleryImage.findById as jest.Mock).mockResolvedValue(doc);
    (deleteImageFromGCS as jest.Mock).mockImplementation(async () => {
      order.push("delete");
      return true;
    });

    await PUT(makeRequest({ image: makeFile() }), makeParams());

    expect(order).toEqual(["save", "delete"]);
  });

  test("does not delete anything when save() fails", async () => {
    const doc = makeDoc(OLD_URL);
    doc.save = jest.fn().mockRejectedValue(new Error("save failed"));
    (GalleryImage.findById as jest.Mock).mockResolvedValue(doc);

    const res: any = await PUT(makeRequest({ image: makeFile() }), makeParams());

    expect(res.status).toBe(500);
    expect(deleteImageFromGCS).not.toHaveBeenCalled();
  });

  test("a title-only update leaves the existing image alone", async () => {
    const doc = makeDoc(OLD_URL);
    (GalleryImage.findById as jest.Mock).mockResolvedValue(doc);

    const res: any = await PUT(makeRequest({ title: "New title" }), makeParams());

    expect(res.status).toBe(200);
    expect(doc.img_url).toBe(OLD_URL);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
    expect(deleteImageFromGCS).not.toHaveBeenCalled();
  });

  test("a legacy Supabase URL is passed through for the helper to skip", async () => {
    const doc = makeDoc(LEGACY_URL);
    (GalleryImage.findById as jest.Mock).mockResolvedValue(doc);
    (deleteImageFromGCS as jest.Mock).mockResolvedValue(false);

    const res: any = await PUT(makeRequest({ image: makeFile() }), makeParams());

    expect(res.status).toBe(200);
    expect(deleteImageFromGCS).toHaveBeenCalledWith(LEGACY_URL, "test-bucket");
  });

  test("a failed cleanup does not fail the request", async () => {
    const doc = makeDoc(OLD_URL);
    (GalleryImage.findById as jest.Mock).mockResolvedValue(doc);
    (deleteImageFromGCS as jest.Mock).mockRejectedValue(new Error("delete failed"));

    const res: any = await PUT(makeRequest({ image: makeFile() }), makeParams());

    expect(res.status).toBe(200);
  });

  test("upload failure leaves the stored URL untouched", async () => {
    const doc = makeDoc(OLD_URL);
    (GalleryImage.findById as jest.Mock).mockResolvedValue(doc);
    (uploadImageToGCS as jest.Mock).mockRejectedValue(new Error("GCS unavailable"));

    const res: any = await PUT(makeRequest({ image: makeFile() }), makeParams());

    expect(res.status).toBe(500);
    expect(doc.img_url).toBe(OLD_URL);
    expect(doc.save).not.toHaveBeenCalled();
    expect(deleteImageFromGCS).not.toHaveBeenCalled();
  });
});

// ============================================================
// PUT — permissions and validation
// ============================================================

describe("PUT /api/gallery/[id] — permissions and validation", () => {
  test("anonymous request is rejected with 401", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(ANON_AUTH);

    const res: any = await PUT(makeRequest({ image: makeFile() }), makeParams());

    expect(res.status).toBe(401);
    expect(GalleryImage.findById).not.toHaveBeenCalled();
  });

  test("non-admin request is rejected with 403", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(USER_AUTH);

    const res: any = await PUT(makeRequest({ image: makeFile() }), makeParams());

    expect(res.status).toBe(403);
  });

  test("malformed id is rejected with 400", async () => {
    const res: any = await PUT(makeRequest({ image: makeFile() }), makeParams("not-an-id"));

    expect(res.status).toBe(400);
    expect(GalleryImage.findById).not.toHaveBeenCalled();
  });

  test("unknown id returns 404", async () => {
    (GalleryImage.findById as jest.Mock).mockResolvedValue(null);

    const res: any = await PUT(makeRequest({ image: makeFile() }), makeParams());

    expect(res.status).toBe(404);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });

  test("oversized replacement image is rejected before upload", async () => {
    (GalleryImage.findById as jest.Mock).mockResolvedValue(makeDoc());

    const res: any = await PUT(
      makeRequest({ image: makeFile({ size: 5 * 1024 * 1024 + 1 }) }),
      makeParams(),
    );

    expect(res.status).toBe(400);
    expect(uploadImageToGCS).not.toHaveBeenCalled();
  });
});

// ============================================================
// DELETE
// ============================================================

describe("DELETE /api/gallery/[id]", () => {
  test("removes the record and its backing object", async () => {
    (GalleryImage.findById as jest.Mock).mockReturnValue({
      lean: jest.fn().mockResolvedValue({ _id: IMAGE_ID, img_url: OLD_URL }),
    });
    (GalleryImage.findByIdAndDelete as jest.Mock).mockResolvedValue({ _id: IMAGE_ID });

    const res: any = await DELETE(makeRequest({}, "DELETE"), makeParams());

    expect(res.status).toBe(200);
    expect(GalleryImage.findByIdAndDelete).toHaveBeenCalledWith(IMAGE_ID);
    expect(deleteImageFromGCS).toHaveBeenCalledWith(OLD_URL, "test-bucket");
  });

  test("unknown id returns 404 and deletes nothing", async () => {
    (GalleryImage.findById as jest.Mock).mockReturnValue({
      lean: jest.fn().mockResolvedValue(null),
    });

    const res: any = await DELETE(makeRequest({}, "DELETE"), makeParams());

    expect(res.status).toBe(404);
    expect(GalleryImage.findByIdAndDelete).not.toHaveBeenCalled();
    expect(deleteImageFromGCS).not.toHaveBeenCalled();
  });

  test("non-admin request is rejected with 403", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(USER_AUTH);

    const res: any = await DELETE(makeRequest({}, "DELETE"), makeParams());

    expect(res.status).toBe(403);
    expect(GalleryImage.findByIdAndDelete).not.toHaveBeenCalled();
  });

  test("storage cleanup failure still reports success", async () => {
    (GalleryImage.findById as jest.Mock).mockReturnValue({
      lean: jest.fn().mockResolvedValue({ _id: IMAGE_ID, img_url: OLD_URL }),
    });
    (GalleryImage.findByIdAndDelete as jest.Mock).mockResolvedValue({ _id: IMAGE_ID });
    (deleteImageFromGCS as jest.Mock).mockRejectedValue(new Error("delete failed"));

    const res: any = await DELETE(makeRequest({}, "DELETE"), makeParams());

    expect(res.status).toBe(200);
  });
});

// ============================================================
// GET
// ============================================================

describe("GET /api/gallery/[id]", () => {
  test("returns the record for an authenticated caller", async () => {
    (GalleryImage.findById as jest.Mock).mockReturnValue({
      select: jest.fn(() => ({
        lean: jest.fn().mockResolvedValue({
          _id: { toString: () => IMAGE_ID },
          title: "Original title",
          img_url: OLD_URL,
        }),
      })),
    });

    const res: any = await GET(makeRequest({}, "GET"), makeParams());

    expect(res.status).toBe(200);
    expect(res._body.data.id).toBe(IMAGE_ID);
  });

  test("anonymous request is rejected with 401", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(ANON_AUTH);

    const res: any = await GET(makeRequest({}, "GET"), makeParams());

    expect(res.status).toBe(401);
  });
});
