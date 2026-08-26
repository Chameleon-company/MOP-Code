/**
 * @jest-environment node
 *
 * Parity tests for the MongoDB /api/contributors/:id routes.
 * All MongoDB/Mongoose calls are mocked — no real DB work happens here.
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

jest.mock("@/models/mongoose/Contributor", () => ({
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
    (
      message: string,
      status: number,
      code: string,
    ) => ({
      status,
      json: jest.fn().mockResolvedValue({
        success: false,
        message,
        code,
      }),
    }),
  ),
}));

jest.mock("@/app/api/library/contributorDto", () => ({
  toContributorDTO: jest.fn((contributor) => ({
    id: contributor._id,
    name: contributor.name,
    year: contributor.year,
    trimester: contributor.trimester,
    contributor_type: contributor.contributor_type,
    team: contributor.team,
    position: contributor.position,
    level: contributor.level,
    display_order: contributor.display_order,
    is_active: contributor.is_active,
  })),
}));

// ==============================
// Imports
// ==============================

import { GET, PUT, DELETE } from "@/app/api/contributors/[id]/route";
import dbConnect from "@/lib/dbConnect";
import Contributor from "@/models/mongoose/Contributor";
import { getAuthUser } from "@/app/api/library/auth";

// ============================================================
// Test helpers
// ============================================================

function makeRequest(
  method = "GET",
  body?: unknown,
) {
  return {
    method,
    json: jest.fn().mockResolvedValue(body),
  } as any;
}

function makeInvalidJsonRequest() {
  return {
    method: "PUT",
    json: jest.fn().mockRejectedValue(
      new SyntaxError("Unexpected token"),
    ),
  } as any;
}

function makeParams(id: string) {
  return {
    params: Promise.resolve({ id }),
  };
}

// ============================================================
// Auth mocks
// ============================================================

const ADMIN_AUTH = {
  userId: 9,
  roleId: 1,
  roleName: "admin",
  isAuthenticated: true,
  isAdmin: true,
};

const USER_AUTH = {
  userId: 7,
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

// ============================================================
// Mock contributor
// ============================================================

const CONTRIBUTOR_ID = "507f1f77bcf86cd799439011";

const MOCK_CONTRIBUTOR = {
  _id: CONTRIBUTOR_ID,
  legacy_id: "1",
  name: "Josh Smith",
  year: 2026,
  trimester: 2,
  contributor_type: "student",
  team: "Data Science Team",
  position: "Project Leader",
  level: "Senior",
  display_order: 1,
  is_active: true,
};

// ==============================
// Tests
// ==============================

// ============================================================
// GET /api/contributors/:id
// ============================================================

describe("GET /api/contributors/:id", () => {
  beforeEach(() => {
    jest.clearAllMocks();

    (dbConnect as jest.Mock).mockResolvedValue(undefined);

    (Contributor.findById as jest.Mock).mockReturnValue({
      lean: jest.fn().mockResolvedValue(MOCK_CONTRIBUTOR),
    });
  });

  test("returns contributor in the frontend-expected response shape", async () => {
    const response = await GET(
      makeRequest(),
      makeParams(CONTRIBUTOR_ID),
    );

    const body = await response.json();

    expect(response.status).toBe(200);

    expect(body).toEqual({
      success: true,
      data: {
        id: CONTRIBUTOR_ID,
        name: "Josh Smith",
        year: 2026,
        trimester: 2,
        contributor_type: "student",
        team: "Data Science Team",
        position: "Project Leader",
        level: "Senior",
        display_order: 1,
        is_active: true,
      },
    });

    expect(dbConnect).toHaveBeenCalled();

    expect(Contributor.findById).toHaveBeenCalledWith(
      CONTRIBUTOR_ID,
    );
  });

  test("returns 400 for an invalid contributor ID", async () => {
    const response = await GET(
      makeRequest(),
      makeParams("invalid-id"),
    );

    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.success).toBe(false);

    expect(dbConnect).not.toHaveBeenCalled();
    expect(Contributor.findById).not.toHaveBeenCalled();
  });

  test("returns 404 when contributor does not exist", async () => {
    (Contributor.findById as jest.Mock).mockReturnValue({
      lean: jest.fn().mockResolvedValue(null),
    });

    const response = await GET(
      makeRequest(),
      makeParams(CONTRIBUTOR_ID),
    );

    const body = await response.json();

    expect(response.status).toBe(404);
    expect(body.success).toBe(false);
    expect(body.message).toBe("Contributor not found");
  });

  test("returns 500 when fetching the contributor fails", async () => {
    (Contributor.findById as jest.Mock).mockReturnValue({
      lean: jest.fn().mockRejectedValue(
        new Error("Database failure"),
      ),
    });

    const response = await GET(
      makeRequest(),
      makeParams(CONTRIBUTOR_ID),
    );

    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.success).toBe(false);
  });
});

// ============================================================
// PUT /api/contributors/:id
// ============================================================

describe("PUT /api/contributors/:id", () => {
  let existingContributor: {
    set: jest.Mock;
    save: jest.Mock;
    toObject: jest.Mock;
  };

  beforeEach(() => {
    jest.clearAllMocks();

    (getAuthUser as jest.Mock).mockReturnValue(ADMIN_AUTH);
    (dbConnect as jest.Mock).mockResolvedValue(undefined);

    existingContributor = {
      set: jest.fn(),
      save: jest.fn().mockResolvedValue(undefined),
      toObject: jest.fn().mockReturnValue(
        MOCK_CONTRIBUTOR,
      ),
    };

    (Contributor.findById as jest.Mock).mockResolvedValue(
      existingContributor,
    );
  });

  test("updates contributor using the frontend payload shape", async () => {
    const payload = {
      name: "John Smith",
      year: 2026,
      trimester: 2,
      contributor_type: "student",
      team: "Development Team",
      position: "Team Leader",
      level: "Senior",
      display_order: 5,
      is_active: true,
    };

    const response = await PUT(
      makeRequest("PUT", payload),
      makeParams(CONTRIBUTOR_ID),
    );

    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.success).toBe(true);

    expect(existingContributor.set).toHaveBeenCalledWith({
      name: "John Smith",
      year: 2026,
      trimester: 2,
      contributor_type: "student",
      team: "Development Team",
      position: "Team Leader",
      level: "Senior",
      display_order: 5,
      is_active: true,
    });

    expect(existingContributor.save).toHaveBeenCalled();

    expect(dbConnect).toHaveBeenCalled();
    expect(Contributor.findById).toHaveBeenCalledWith(
      CONTRIBUTOR_ID,
    );
  });

  test("trims name, team, and position before saving", async () => {
    const payload = {
      name: "  Josh Smith  ",
      year: 2026,
      trimester: 2,
      contributor_type: "student",
      team: "  Development Team  ",
      position: "  Team Leader  ",
      level: "Senior",
      display_order: 5,
      is_active: true,
    };

    await PUT(
      makeRequest("PUT", payload),
      makeParams(CONTRIBUTOR_ID),
    );

    expect(existingContributor.set).toHaveBeenCalledWith({
      name: "Josh Smith",
      year: 2026,
      trimester: 2,
      contributor_type: "student",
      team: "Development Team",
      position: "Team Leader",
      level: "Senior",
      display_order: 5,
      is_active: true,
    });
  });

  test("clears student-only fields for a mentor", async () => {
    const payload = {
      name: "Josh Smith",
      year: 2026,
      trimester: 2,
      contributor_type: "mentor",
      team: "Should be removed",
      position: "Project Mentor",
      level: "Senior",
      display_order: 3,
      is_active: true,
    };

    await PUT(
      makeRequest("PUT", payload),
      makeParams(CONTRIBUTOR_ID),
    );

    expect(existingContributor.set).toHaveBeenCalledWith(
      expect.objectContaining({
        contributor_type: "mentor",
        team: null,
        position: "Project Mentor",
        level: null,
      }),
    );
  });

  test("clears student-only fields for a company director", async () => {
    const payload = {
      name: "Josh Smith",
      year: 2026,
      trimester: 2,
      contributor_type: "company_director",
      team: "Should be removed",
      position: "Should be removed",
      level: "Senior",
      display_order: 5,
      is_active: true,
    };

    await PUT(
      makeRequest("PUT", payload),
      makeParams(CONTRIBUTOR_ID),
    );

    expect(existingContributor.set).toHaveBeenCalledWith(
      expect.objectContaining({
        contributor_type: "company_director",
        team: null,
        position: null,
        level: null,
      }),
    );
  });

  test("returns updated contributor in the frontend-expected response shape", async () => {
    const updatedContributor = {
      ...MOCK_CONTRIBUTOR,
      trimester: 2,
      team: "Development Team",
      position: "Team Leader",
    };

    existingContributor.toObject.mockReturnValue(
      updatedContributor,
    );

    const payload = {
      name: "Josh Smith",
      year: 2026,
      trimester: 2,
      contributor_type: "student",
      team: "Development Team",
      position: "Team Leader",
      level: "Senior",
      display_order: 5,
      is_active: true,
    };

    const response = await PUT(
      makeRequest("PUT", payload),
      makeParams(CONTRIBUTOR_ID),
    );

    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.data).toEqual(
      expect.objectContaining({
        id: CONTRIBUTOR_ID,
        name: "Josh Smith",
        year: 2026,
        trimester: 2,
        contributor_type: "student",
        team: "Development Team",
        position: "Team Leader",
        level: "Senior",
        display_order: 1,
        is_active: true,
      }),
    );
  });

  test("returns 401 for an unauthenticated user", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(
      ANON_AUTH,
    );

    const response = await PUT(
      makeRequest("PUT", {}),
      makeParams(CONTRIBUTOR_ID),
    );

    const body = await response.json();

    expect(response.status).toBe(401);
    expect(body.success).toBe(false);

    expect(dbConnect).not.toHaveBeenCalled();
    expect(Contributor.findById).not.toHaveBeenCalled();
  });

  test("returns 403 for an authenticated non-admin user", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(
      USER_AUTH,
    );

    const response = await PUT(
      makeRequest("PUT", {}),
      makeParams(CONTRIBUTOR_ID),
    );

    const body = await response.json();

    expect(response.status).toBe(403);
    expect(body.success).toBe(false);

    expect(dbConnect).not.toHaveBeenCalled();
    expect(Contributor.findById).not.toHaveBeenCalled();
  });

  test("returns 400 for an invalid contributor ID", async () => {
    const response = await PUT(
      makeRequest("PUT", {}),
      makeParams("invalid-id"),
    );

    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.success).toBe(false);

    expect(dbConnect).not.toHaveBeenCalled();
    expect(Contributor.findById).not.toHaveBeenCalled();
  });

  test("returns 400 for invalid JSON", async () => {
    const response = await PUT(
      makeInvalidJsonRequest(),
      makeParams(CONTRIBUTOR_ID),
    );

    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.success).toBe(false);
    expect(body.message).toBe(
      "Invalid JSON in request body.",
    );

    expect(dbConnect).not.toHaveBeenCalled();
  });

  test("returns 404 when contributor does not exist", async () => {
    (Contributor.findById as jest.Mock).mockResolvedValue(
      null,
    );

    const response = await PUT(
      makeRequest("PUT", {
        name: "Josh Smith",
        year: 2026,
        trimester: 2,
        contributor_type: "student",
        team: "Development Team",
        position: "Team Leader",
        level: "Senior",
        display_order: 5,
        is_active: true,
      }),
      makeParams(CONTRIBUTOR_ID),
    );

    const body = await response.json();

    expect(response.status).toBe(404);
    expect(body.success).toBe(false);
    expect(body.message).toBe(
      "Contributor not found",
    );
  });

  test("returns 400 for a Mongoose validation error", async () => {
    const validationError = new Error(
      "Contributor validation failed",
    );

    validationError.name = "ValidationError";

    existingContributor.save.mockRejectedValue(
      validationError,
    );

    const response = await PUT(
      makeRequest("PUT", {
        name: "Josh Smith",
        year: 2026,
        trimester: 2,
        contributor_type: "student",
        team: "Development Team",
        position: "Team Leader",
        level: "Senior",
        display_order: 5,
        is_active: true,
      }),
      makeParams(CONTRIBUTOR_ID),
    );

    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.success).toBe(false);
    expect(body.message).toBe(
      "Contributor validation failed",
    );
  });

  test("returns 500 when updating the contributor fails", async () => {
    existingContributor.save.mockRejectedValue(
      new Error("Database failure"),
    );

    const response = await PUT(
      makeRequest("PUT", {
        name: "Josh Smith",
        year: 2026,
        trimester: 2,
        contributor_type: "student",
        team: "Development Team",
        position: "Team Leader",
        level: "Senior",
        display_order: 5,
        is_active: true,
      }),
      makeParams(CONTRIBUTOR_ID),
    );

    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.success).toBe(false);
  });
});

// ============================================================
// DELETE /api/contributors/:id
// ============================================================

describe("DELETE /api/contributors/:id", () => {
  beforeEach(() => {
    jest.clearAllMocks();

    (getAuthUser as jest.Mock).mockReturnValue(
      ADMIN_AUTH,
    );

    (dbConnect as jest.Mock).mockResolvedValue(undefined);

    (Contributor.findByIdAndDelete as jest.Mock).mockResolvedValue(
      MOCK_CONTRIBUTOR,
    );
  });

  test("deletes contributor successfully", async () => {
    const response = await DELETE(
      makeRequest("DELETE"),
      makeParams(CONTRIBUTOR_ID),
    );

    const body = await response.json();

    expect(response.status).toBe(200);

    expect(body).toEqual({
      success: true,
      message: "Contributor deleted successfully",
    });

    expect(dbConnect).toHaveBeenCalled();

    expect(
      Contributor.findByIdAndDelete,
    ).toHaveBeenCalledWith(CONTRIBUTOR_ID);
  });

  test("returns 401 for an unauthenticated user", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(
      ANON_AUTH,
    );

    const response = await DELETE(
      makeRequest("DELETE"),
      makeParams(CONTRIBUTOR_ID),
    );

    const body = await response.json();

    expect(response.status).toBe(401);
    expect(body.success).toBe(false);

    expect(dbConnect).not.toHaveBeenCalled();
    expect(
      Contributor.findByIdAndDelete,
    ).not.toHaveBeenCalled();
  });

  test("returns 403 for an authenticated non-admin user", async () => {
    (getAuthUser as jest.Mock).mockReturnValue(
      USER_AUTH,
    );

    const response = await DELETE(
      makeRequest("DELETE"),
      makeParams(CONTRIBUTOR_ID),
    );

    const body = await response.json();

    expect(response.status).toBe(403);
    expect(body.success).toBe(false);

    expect(dbConnect).not.toHaveBeenCalled();
    expect(
      Contributor.findByIdAndDelete,
    ).not.toHaveBeenCalled();
  });

  test("returns 400 for an invalid contributor ID", async () => {
    const response = await DELETE(
      makeRequest("DELETE"),
      makeParams("invalid-id"),
    );

    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.success).toBe(false);

    expect(dbConnect).not.toHaveBeenCalled();
    expect(
      Contributor.findByIdAndDelete,
    ).not.toHaveBeenCalled();
  });

  test("returns 404 when contributor does not exist", async () => {
    (Contributor.findByIdAndDelete as jest.Mock).mockResolvedValue(
      null,
    );

    const response = await DELETE(
      makeRequest("DELETE"),
      makeParams(CONTRIBUTOR_ID),
    );

    const body = await response.json();

    expect(response.status).toBe(404);
    expect(body.success).toBe(false);
    expect(body.message).toBe(
      "Contributor not found",
    );
  });

  test("returns 500 when deletion fails", async () => {
    (Contributor.findByIdAndDelete as jest.Mock).mockRejectedValue(
      new Error("Database failure"),
    );

    const response = await DELETE(
      makeRequest("DELETE"),
      makeParams(CONTRIBUTOR_ID),
    );

    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.success).toBe(false);
  });
});