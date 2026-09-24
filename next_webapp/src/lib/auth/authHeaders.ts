/**
 * Builds the request headers admin API routes expect for authentication
 * and role-based authorization. The backend (see getAuthUser in
 * src/app/api/library/auth.ts) derives isAuthenticated/isAdmin from the
 * x-user-* headers, not from the Authorization header, so all four must
 * be sent together.
 */
export function getAuthHeaders(): Record<string, string> {
  if (typeof window === "undefined") return {};

  let user: Record<string, any> = {};
  try {
    user = JSON.parse(window.localStorage.getItem("user") || "{}");
  } catch {
    user = {};
  }

  const userId = user.userId ?? user.id ?? window.localStorage.getItem("userId") ?? "";
  const roleId = user.roleId ?? user.role_id ?? "";
  const token = user.token ?? "";

  return {
    "x-user-id": String(userId),
    "x-user-role-id": String(roleId),
    "x-user-role": user.roleName ?? user.role_name ?? "",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}
