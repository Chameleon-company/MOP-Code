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
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
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
    <div className="flex min-h-screen w-full max-w-full overflow-x-hidden bg-[#F5F5F5] text-black">
      <AdminSidebar
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
        mobileMenuOpen={mobileMenuOpen}
        setMobileMenuOpen={setMobileMenuOpen}
      />
      <div className="flex min-w-0 flex-1 flex-col overflow-x-hidden">
        <AdminHeader
          onToggleMobileMenu={() => setMobileMenuOpen((prev) => !prev)}
        />
        <main className="min-w-0 flex-1 p-3.5 sm:p-5 lg:p-8">{children}</main>
      </div>
    </div>
  );
}