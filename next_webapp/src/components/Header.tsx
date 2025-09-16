"use client";
import React, { useState } from "react";
import { useTranslations } from "next-intl";
import { Link } from "@/i18n-navigation";
import LanguageDropdown from "./LanguageDropdown";
import { HiMenu, HiX, HiMoon, HiSun } from "react-icons/hi";
import { useTheme } from "../hooks/useTheme";

const Header = () => {
  const t = useTranslations("common");
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const { theme, toggleTheme } = useTheme();

  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);

  // Navigation items
  const navItems = [
    { name: "Home", link: "/" },
    { name: "About Us", link: "/about" },
    { name: "Use Cases", link: "/usecases" },
    { name: "Statistics", link: "/statistics" },
    { name: "Upload", link: "/upload" },
    { name: "Blogs", link: "/blog" },
  ];

  return (
    <header className="bg-white shadow-sm dark:bg-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            {/* Hamburger menu (mobile) */}
            <div className="flex lg:hidden mr-4">
              <button
                onClick={toggleMenu}
                className="text-green-600 hover:text-green-900 focus:outline-none focus:text-green-900"
              >
                {isMenuOpen ? <HiX className="h-6 w-6" /> : <HiMenu className="h-6 w-6" />}
              </button>
            </div>

            {/* Logo */}
            <Link href="/" className="flex-shrink-0 flex items-center">
              <img
                className="h-20 w-auto"
                src="/img/new-logo-green.png"
                alt="Logo"
              />
              <span className="hidden lg:block ml-2 text-xl font-bold text-green-600 dark:text-green-300">
                Chameleon
              </span>
            </Link>
          </div>

          <div className="flex items-center">
            {/* Desktop Menu */}
            <nav className="hidden lg:flex ml-10 space-x-4">
              {navItems.map((item) => (
                <Link
                  key={item.name}
                  href={item.link}
                  className="text-gray-600 hover:text-green-900 dark:text-gray-200 dark:hover:text-green-300 px-3 py-2 rounded-md text-sm font-medium"
                >
                  {t(item.name)}
                </Link>
              ))}
            </nav>

            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              aria-label="Toggle Dark Mode"
              className="p-1 rounded focus:outline-none ml-4"
            >
              {theme === "dark" ? (
                <HiSun className="h-5 w-5 text-white" />
              ) : (
                <HiMoon className="h-5 w-5 text-black" />
              )}
            </button>

            {/* Language Dropdown */}
            <LanguageDropdown />

            {/* Sign Up / Log In (desktop) */}
            <div className="hidden lg:flex ml-4 space-x-2">
              <Link
                href="/signup"
                className="text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-base font-medium"
              >
                {t("Sign Up")}
              </Link>
              <Link
                href="/login"
                className="text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-base font-medium"
              >
                {t("Log In")}
              </Link>
            </div>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMenuOpen && (
          <div className="lg:hidden mt-2">
            <nav className="px-2 pt-2 pb-3 space-y-1">
              {navItems.map((item) => (
                <Link
                  key={item.name}
                  href={item.link}
                  className="block text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-base font-medium"
                >
                  {t(item.name)}
                </Link>
              ))}
              <Link
                href="/signup"
                className="block text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-base font-medium"
              >
                {t("Sign Up")}
              </Link>
              <Link
                href="/login"
                className="block text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-base font-medium"
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
