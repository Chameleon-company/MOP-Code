"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useLocale } from "next-intl";
import AdminSidebar from "@/components/admin/AdminSidebar";
import AdminHeader from "@/components/admin/AdminHeader";

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [authorized, setAuthorized] = useState(false);
  const router = useRouter();
  const locale = useLocale();

  useEffect(() => {
    const stored = localStorage.getItem("user");
    const user = stored ? JSON.parse(stored) : null;

    if (!user || !user.token) {
      router.replace(`/${locale}/login`);
      return;
    }

    if (user.roleId !== 1) {
      router.replace(`/${locale}/profile`);
      return;
    }

    setAuthorized(true);
  }, []);

  if (!authorized) return null;

  return (
    <div className="flex min-h-screen bg-[#F5F5F5]">
      <AdminSidebar sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
      <div className="flex-1">
        <AdminHeader />
        <main className="p-8">{children}</main>
      </div>
    </div>
  );
}