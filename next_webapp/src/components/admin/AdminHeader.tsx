"use client";

import { Bell, Search, UserCircle2, Settings, LogOut } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import Image from "next/image";

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
    router.push(`/${locale}/login`);
  };

  return (
    <header className="flex h-[72px] w-full items-center justify-between border-b border-[#D9D9D9] bg-[#F1EFEF] px-3 sm:px-4 md:px-6">
      <div className="flex shrink-0 items-center">
        <Image
          src="/img/new-logo-green.png"
          alt="Chameleon Logo"
          width={60}
          height={24}
          className="object-contain sm:w-[72px] md:w-[90px]"
          priority
        />
      </div>

      <div className="mx-2 flex w-full min-w-0 max-w-[160px] items-center rounded-lg border border-[#D9D9D9] bg-white px-3 py-2 sm:mx-4 sm:max-w-[260px] md:max-w-[420px]">
        <Search size={16} className="shrink-0 text-[#9CA3AF]" />
        <input
          type="text"
          placeholder="Search data"
          className="ml-2 w-full min-w-0 bg-transparent text-sm outline-none"
        />
      </div>

      <div className="flex shrink-0 items-center gap-3 sm:gap-4 md:gap-5">
        <button
          type="button"
          className="text-[#4ADE80] transition hover:scale-105"
        >
          <Bell size={20} className="sm:h-[22px] sm:w-[22px]" />
        </button>

        <div className="relative" ref={dropdownRef}>
          <button
            type="button"
            onClick={() => setProfileOpen((prev) => !prev)}
            className="text-[#4ADE80] transition hover:scale-105"
          >
            <UserCircle2 size={26} className="sm:h-[30px] sm:w-[30px]" />
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