"use client";

import { Bell, Search, UserCircle2, Settings, LogOut } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";

export default function AdminHeader() {
  const [profileOpen, setProfileOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement | null>(null);
  const params = useParams();
  const router = useRouter();
  const locale = params?.locale as string;

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setProfileOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const handleLogout = () => {
    setProfileOpen(false);

    // later you can also clear token/cookies here
    // localStorage.removeItem("token");

    router.push(`/${locale}/login`);
  };

  return (
    <header className="flex items-center justify-between border-b border-[#D9D9D9] bg-white px-6 py-4">
      <div className="text-sm font-semibold text-[#1F8F50]">
        Melbourne Open Data
      </div>

      <div className="flex w-full max-w-[420px] items-center rounded-lg border border-[#D9D9D9] bg-[#F8F8F8] px-3 py-2">
        <Search size={16} className="text-[#9CA3AF]" />
        <input
          type="text"
          placeholder="Search data"
          className="ml-2 w-full bg-transparent text-sm outline-none"
        />
      </div>

      <div className="flex items-center gap-4">
        <button
          type="button"
          className="text-[#4ADE80] transition hover:scale-105"
        >
          <Bell size={18} />
        </button>

        <div className="relative" ref={dropdownRef}>
          <button
            type="button"
            onClick={() => setProfileOpen((prev) => !prev)}
            className="text-[#4ADE80] transition hover:scale-105"
          >
            <UserCircle2 size={24} />
          </button>

          {profileOpen && (
            <div className="absolute right-0 top-10 z-50 w-44 rounded-xl border border-[#E5E7EB] bg-white p-2 shadow-lg">
              <Link
                href={`/${locale}/admin/settings`}
                onClick={() => setProfileOpen(false)}
                className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-[#1A1A1A] transition hover:bg-[#F3F4F6]"
              >
                <Settings size={16} />
                Settings
              </Link>

              <button
                type="button"
                onClick={handleLogout}
                className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-red-600 transition hover:bg-red-50"
              >
                <LogOut size={16} />
                Logout
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}