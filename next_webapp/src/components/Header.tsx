"use client";
import React, { useState } from "react";
import { useTranslations } from "next-intl";
import { Link } from "@/i18n-navigation";
import LanguageDropdown from "./LanguageDropdown";
import { HiMenu, HiX, HiMoon, HiSun } from "react-icons/hi";
import { useTheme } from "../hooks/useTheme";
import { usePathname } from "next/navigation";

const Header = () => {
  const t = useTranslations("");
  const pathname = usePathname();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const { theme, toggleTheme } = useTheme();

  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);

  const navItems = [
    { name: "HOME", link: "/" },
    { name: "ABOUT US", link: "/about" },
    { name: "USE CASES", link: "/usecases" },
    { name: "STATISTICS", link: "/statistics" },
    { name: "UPLOAD", link: "/upload" },
  ];

  return (
    <header className="bg-white dark:bg-black transition-colors duration-300 shadow-sm font-[Montserrat]">
      <link
        href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap"
        rel="stylesheet"
      ></link>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center w-full">
            <Link href="/" className="flex-shrink-0">
              <img
                className="h-20 w-auto"
                src="/img/new-logo-green.png"
                alt="Logo"
              />
            </Link>

            {/* Desktop Menu */}
            <nav className="ml-10 gap-x-6 hidden lg:flex">
              {navItems.map((item) => (
                <Link
                  key={item.name}
                  href={item.link}
                  className={`px-3 py-2 rounded-md text-sm font-bold transition-colors duration-300 ${
                    pathname === item.link
                      ? "text-green-700 dark:text-green-400 font-semibold"
                      : "text-gray-800 hover:text-green-900 dark:text-gray-200 dark:hover:text-green-700"
                  }`}
                >
                  {t(item.name)}
                </Link>
              ))}
            </nav>

            {/* Hamburger Icon */}
            <div className="flex lg:hidden ml-auto">
              <button
                onClick={toggleMenu}
                className="text-green-600 hover:text-green-900 focus:outline-none"
              >
                {isMenuOpen ? (
                  <HiX className="h-6 w-6" />
                ) : (
                  <HiMenu className="h-6 w-6" />
                )}
              </button>
            </div>
          </div>

          {/* Right Side Controls */}
          <div className="flex items-center space-x-3 ml-4">
            <button
              onClick={toggleTheme}
              aria-label="Toggle Dark Mode"
              className="p-1 rounded transition-colors duration-300 focus:outline-none"
            >
              {theme === "dark" ? (
                <HiSun className="h-5 w-5 text-white" />
              ) : (
                <HiMoon className="h-5 w-5 text-black" />
              )}
            </button>

            <LanguageDropdown />

            <div className="hidden lg:flex">
              <Link
                href="/signup"
                className="bg-green-600 text-white hover:bg-green-700 px-8 py-2 rounded-xl text-sm font-semibold shadow"
              >
                {t("CREATE ACCOUNT")}
              </Link>
              <Link
                href="/login"
                className="ml-4 bg-white text-green-600 hover:bg-gray-100 border border-green-600 px-10 py-2 rounded-xl text-sm font-semibold"
              >
                {t("LOGIN")}
              </Link>
            </div>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMenuOpen && (
          <div className="lg:hidden">
            <nav className="px-4 pt-4 pb-6 space-y-2 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 shadow-md rounded-md">
              {navItems.map((item) => (
                <Link
                  key={item.name}
                  href={item.link}
                  className="block px-3 py-2 rounded-md text-base font-medium text-green-600 hover:text-green-800 transition-colors duration-300"
                >
                  {t(item.name)}
                </Link>
              ))}
              <Link
                href="/signup"
                className="block px-3 py-2 rounded-md text-base font-medium text-green-600 hover:text-green-800"
              >
                {t("Sign Up")}
              </Link>
              <Link
                href="/login"
                className="block px-3 py-2 rounded-md text-base font-medium text-green-600 hover:text-green-800"
              >
                {t("Log In")}
              </Link>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;
