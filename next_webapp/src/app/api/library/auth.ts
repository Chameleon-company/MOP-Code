import { NextRequest } from "next/server";

export function getAuthUser(request: NextRequest) {
  const userIdRaw = request.headers.get("x-user-id");
  const roleIdRaw = request.headers.get("x-user-role-id");
  const roleName = request.headers.get("x-user-role");

  const userId = userIdRaw ? Number(userIdRaw) : null;
  const roleId = roleIdRaw ? Number(roleIdRaw) : null;

  return {
    userId,
    roleId,
    roleName,
    isAuthenticated: !!userId,
    isAdmin: roleId === 1,
  };
}