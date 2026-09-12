"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useLocale } from "next-intl";
import { Menu } from "lucide-react";
import AdminSidebar from "@/components/admin/AdminSidebar";
import AdminHeader from "@/components/admin/AdminHeader";

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileOpen, setMobileOpen] = useState(false);
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
      <AdminSidebar
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
        mobileOpen={mobileOpen}
        setMobileOpen={setMobileOpen}
      />
      <div className="flex-1">
        <div className="flex items-center border-b border-[#D9D9D9] bg-[#F1EFEF] px-3 py-2 lg:hidden">
          <button
            type="button"
            onClick={() => setMobileOpen(true)}
            aria-label="Open menu"
            className="flex h-10 w-10 items-center justify-center rounded-lg text-black transition hover:bg-black/5"
          >
            <Menu size={20} />
          </button>
        </div>
        <AdminHeader />
        <main className="p-8">{children}</main>
      </div>
    </div>
  );
}