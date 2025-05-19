//component/Header.tsx
'use client';

import React, { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { Link } from "@/i18n-navigation";
import LanguageDropdown from "./LanguageDropdown";
import { HiMenu, HiX } from "react-icons/hi";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faSun, faMoon } from "@fortawesome/free-solid-svg-icons";
import { useRouter } from "next/navigation";

const navItems = [
  { name: "Home", href: "/" },
  { name: "About Us", href: "/about" },
  { name: "Use Cases", href: "/usecases" },
  { name: "Statistics", href: "/statistics" },
  { name: "Upload", href: "/upload" },
];

const Header: React.FC = () => {
  const t = useTranslations("common");
  const router = useRouter();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [user, setUser] = useState<any>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);

  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  const toggleMenu = () => setIsMenuOpen((prev) => !prev);
  const toggleDarkMode = () => setDarkMode((prev) => !prev);

  const handleLogout = () => {
    localStorage.removeItem("user");
    setUser(null);
    router.push("/en/login");
  };

  return (
    <header className="bg-white shadow-sm dark:bg-gray-900 dark:text-white border">
      {/* Google Fonts */}
      <link
        href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap"
        rel="stylesheet"
      />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Logo and Desktop Navigation */}
        <div className="flex items-center space-x-8">
          <Link href="/" className="flex-shrink-0">
            <img
              src="/img/new-logo-green.png"
              alt="Logo"
              className="h-20 w-auto"
            />
          </Link>
          <nav className="hidden lg:flex space-x-4">
            {navItems.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                className="text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-sm font-medium"
              >
                {t(item.name)}
              </Link>
            ))}
          </nav>
        </div>

        {/* Right-side actions */}
        <div className="flex items-center space-x-4">
          <button
            onClick={toggleDarkMode}
            className="p-1 focus:outline-none"
            aria-label="Toggle dark mode"
          >
            {darkMode ? (
              <FontAwesomeIcon
                icon={faSun}
                size="lg"
                className="text-yellow-400"
              />
            ) : (
              <FontAwesomeIcon
                icon={faMoon}
                size="lg"
                className="text-blue-800"
              />
            )}
          </button>
          <LanguageDropdown />
          {user ? (
            <>
              <span className="text-sm text-gray-700">Hi, {user.name}</span>
              <button
                onClick={handleLogout}
                className="bg-red-500 text-white hover:bg-red-600 px-4 py-2 rounded-md text-sm font-medium"
              >
                Logout
              </button>
            </>
          ) : (
            <>
              <Link
                href="/signup"
                className="bg-green-600 text-white hover:bg-green-700 px-4 py-2 rounded-md text-sm font-medium"
              >
                {t("Sign Up")}
              </Link>
              <Link
                href="/login"
                className="ml-4 bg-white text-green-600 hover:bg-gray-50 border border-green-600 px-4 py-2 rounded-md text-sm font-medium"
              >
                {t("Log In")}
              </Link>
            </>
          )}
          {/* Mobile menu toggle */}
          <button
            onClick={toggleMenu}
            className="lg:hidden text-green-600 hover:text-green-900 focus:outline-none"
            aria-label="Toggle menu"
          >
            {isMenuOpen ? <HiX size={24} /> : <HiMenu size={24} />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation */}
      {isMenuOpen && (
        <div className="lg:hidden bg-white dark:bg-gray-900 px-4 pb-4">
          <nav className="space-y-1">
            {navItems.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                className="block text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-base font-medium"
              >
                {t(item.name)}
              </Link>
            ))}
            {user ? (
              <>
                <span className="block text-green-800 px-3 py-2 text-base font-medium">
                  Hi, {user.name}
                </span>
                <button
                  onClick={handleLogout}
                  className="block w-full text-left text-red-600 hover:text-red-800 px-3 py-2 rounded-md text-base font-medium"
                >
                  Logout
                </button>
              </>
            ) : (
              <>
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
              </>
            )}
          </nav>
        </div>
      )}
    </header>
  );
};

export default Header;