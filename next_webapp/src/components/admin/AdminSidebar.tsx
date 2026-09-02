"use client";

import { useEffect } from "react";
import Link from "next/link";
import { useParams, usePathname } from "next/navigation";
import {
  Menu,
  X,
  LayoutDashboard,
  FolderOpen,
  Briefcase,
  Image as ImageIcon,
  FileText,
  History,
  Users,
} from "lucide-react";
import Image from "next/image";

const menuItems = [
  { label: "Dashboard", path: "/admin/dashboard", icon: LayoutDashboard },
  { label: "Categories", path: "/admin/categories", icon: FolderOpen },
  { label: "Use Cases", path: "/admin/use-cases", icon: Briefcase },
  { label: "Gallery", path: "/admin/gallery", icon: ImageIcon },
  { label: "Contributors", path: "/admin/contributors", icon: Users },
  { label: "Activity History", path: "/admin/activity-history", icon: History },
  { label: "Blogs", path: "/admin/blogs", icon: FileText },
];

type AdminSidebarProps = {
  sidebarOpen: boolean;
  setSidebarOpen: React.Dispatch<React.SetStateAction<boolean>>;
  mobileMenuOpen?: boolean;
  setMobileMenuOpen?: React.Dispatch<React.SetStateAction<boolean>>;
};

export default function AdminSidebar({
  sidebarOpen,
  setSidebarOpen,
  mobileMenuOpen = false,
  setMobileMenuOpen,
}: AdminSidebarProps) {
  const pathname = usePathname();
  const params = useParams();
  const locale = params?.locale as string;

  const closeMobileMenu = () => {
    if (setMobileMenuOpen) {
      setMobileMenuOpen(false);
    }
  };

  useEffect(() => {
    if (!mobileMenuOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        closeMobileMenu();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [mobileMenuOpen]);

  return (
    <>
      {/* ── Mobile Off-Canvas Drawer (< 1024px) ── */}
      {mobileMenuOpen && (
        <div
          role="button"
          tabIndex={0}
          aria-label="Close menu backdrop"
          onClick={closeMobileMenu}
          className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm transition-opacity lg:hidden"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-64 flex-col bg-[#1F8F50] shadow-2xl transition-transform duration-300 ease-in-out lg:hidden ${mobileMenuOpen ? "translate-x-0" : "-translate-x-full"
          }`}
      >
        <div className="flex h-[72px] items-center justify-between border-b border-white/15 px-4">
          <div className="flex items-center gap-2">
            <Image
              src="/img/new-logo-green.png"
              alt="Logo"
              width={70}
              height={24}
              className="brightness-0 invert object-contain"
            />
            <span className="text-xs font-semibold uppercase tracking-wider text-white/90">
              Admin
            </span>
          </div>
          <button
            type="button"
            onClick={closeMobileMenu}
            aria-label="Close menu"
            className="flex h-9 w-9 items-center justify-center rounded-lg text-white transition hover:bg-white/20"
          >
            <X size={20} />
          </button>
        </div>

        <nav className="flex-1 space-y-1.5 overflow-y-auto px-3 py-4">
          {menuItems.map((item) => {
            const href = `/${locale}${item.path}`;
            const isActive = pathname === href;
            const Icon = item.icon;

            return (
              <Link
                key={item.label}
                href={href}
                onClick={closeMobileMenu}
                className={`flex items-center gap-3 rounded-xl px-3.5 py-2.5 text-sm font-medium transition-all duration-200 ${isActive
                    ? "bg-white text-[#1F8F50] shadow-sm"
                    : "text-white/90 hover:bg-white/15 hover:text-white"
                  }`}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </aside>

      {/* ── Desktop Collapsible Sidebar (>= 1024px) ── */}
      <aside
        className={`hidden shrink-0 transition-all duration-300 shadow-sm lg:flex lg:flex-col ${sidebarOpen ? "w-[190px] bg-[#1F8F50]" : "w-[70px] bg-[#F1EFEF]"
          }`}
      >
        <div className="flex h-[72px] items-center px-3">
          <button
            type="button"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
            className={`flex h-10 w-10 items-center justify-center rounded-lg transition ${sidebarOpen
                ? "text-white hover:bg-white/20"
                : "text-black hover:bg-black/5"
              }`}
          >
            <Menu size={20} />
          </button>
        </div>

        <nav className="space-y-2 px-2 pt-2">
          {menuItems.map((item) => {
            const href = `/${locale}${item.path}`;
            const isActive = pathname === href;
            const Icon = item.icon;

            return (
              <Link
                key={item.label}
                href={href}
                title={!sidebarOpen ? item.label : ""}
                className={`flex items-center rounded-lg transition-all duration-200 ${sidebarOpen ? "gap-3 px-3 py-2" : "justify-center px-0 py-2"
                  } ${isActive
                    ? "bg-white text-[#1F8F50]"
                    : sidebarOpen
                      ? "text-white hover:bg-white/20"
                      : "text-black hover:bg-black/5"
                  }`}
              >
                <Icon size={16} className="h-[18px] w-[18px]" />
                {sidebarOpen && (
                  <span className="text-[14px] font-medium">{item.label}</span>
                )}
              </Link>
            );
          })}
        </nav>
      </aside>
    </>
  );
}