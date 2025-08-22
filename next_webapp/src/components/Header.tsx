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

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  // Updated navigation items to match wireframe
  const navItems = [
    { name: "Dashboard", link: "/dashboard" },
    { name: "Organisations", link: "/organisations" },
    { name: "About", link: "/about" },
    { name: "Home", link: "/" },
  ];

  return (
    <header className="bg-white shadow-sm dark:bg-black">
      <link
        href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap"
        rel="stylesheet"
      ></link>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            {/* Hamburger Menu Icon - Moved to left */}
            <div className="flex lg:hidden mr-4">
              <button
                onClick={toggleMenu}
                className="text-green-600 hover:text-green-900 focus:outline-none focus:text-green-900"
              >
                {isMenuOpen ? (
                  <HiX className="h-6 w-6" />
                ) : (
                  <HiMenu className="h-6 w-6" />
                )}
              </button>
            </div>
            
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
            {/* Desktop Menu Items */}
            <nav className="hidden lg:flex ml-10 space-x-4">
              {navItems.map((item) => (
                <Link
                  key={item.name}
                  href={item.link}
                  className="text-black-600 hover:text-green-900 dark:text-gray-200 dark:hover:text-green-300 px-3 py-2 rounded-md text-sm font-medium"
                >
                  {item.name}
                </Link>
              ))}
            </nav>

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

            <Link
              href="/signup"
              className="hidden lg:block ml-4 text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-base font-medium"
            >
              {t("Sign Up")}
            </Link>
          </div>
        </div>
        
        {/* Mobile Menu */}
        {isMenuOpen && (
          <div className="lg:hidden">
            <nav className="px-2 pt-2 pb-3 space-y-1">
              {navItems.map((item) => (
                <Link
                  key={item.name}
                  href={item.link}
                  className="block text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-base font-medium"
                >
                  {item.name}
                </Link>
              ))}
              <Link
                href="/signup"
                className="block text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-base font-medium"
              >
                {t("Sign Up")}
              </Link>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;