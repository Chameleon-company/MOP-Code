'use client'
import React, { useState } from 'react';
import { useTranslations } from 'next-intl';
import { Link } from '@/i18n-navigation';
import LanguageDropdown from './LanguageDropdown';
import { HiMenu, HiX } from 'react-icons/hi';

<<<<<<< Updated upstream
const Header = () => {
  const t = useTranslations('common');
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  // Object array for navigation items
=======
import React, { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { Link } from "@/i18n-navigation";
import LanguageDropdown from "./LanguageDropdown";
import { HiMenu, HiX } from "react-icons/hi";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faSun, faMoon } from "@fortawesome/free-solid-svg-icons";
import { useRouter } from "next/navigation";

const Header: React.FC = () => {
  const t = useTranslations("common");
  const router = useRouter();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [user, setUser] = useState<{ name: string } | null>(null);

  // apply dark class
  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);

  // load user from localStorage
  useEffect(() => {
    const stored = localStorage.getItem("user");
    if (stored) setUser(JSON.parse(stored));
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("user");
    setUser(null);
    router.push("/en/login");
  };

>>>>>>> Stashed changes
  const navItems = [
    { name: 'Home', link: '/' },
    { name: 'About Us', link: '/about' },
    { name: 'Use Cases', link: '/UseCases' },
    { name: 'Statistics', link: '/statistics' },
    { name: 'Upload', link: '/upload' }
  ];

  const toggleMenu = () => setIsMenuOpen((o) => !o);
  const toggleDarkMode = () => setDarkMode((d) => !d);

  return (
<<<<<<< Updated upstream
    <header className="bg-white shadow-sm">
       <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet"></link>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <Link href="/" className="flex-shrink-0">
              <img className="h-20 w-auto" src="/img/new-logo-green.png" alt="Logo" />
            </Link>
            {/* Hamburger Menu Icon */}
            <div className="flex lg:hidden ml-auto">
=======
    <header className="bg-white shadow-sm dark:bg-gray-900 dark:text-white">
      {/* load font */}
      <link
        href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap"
        rel="stylesheet"
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* left: logo + desktop nav + mobile icons */}
        <div className="flex items-center space-x-8">
          {/* Logo */}
          <Link href="/" className="flex-shrink-0">
            <img
              src="/img/new-logo-green.png"
              alt="Logo"
              className="h-20 w-auto"
            />
          </Link>

          {/* Desktop navigation */}
          <nav className="hidden lg:flex space-x-4">
            {navItems.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                className="text-black hover:text-green-900 dark:text-green-300 dark:hover:text-white px-3 py-2 text-sm font-medium"
              >
                {t(item.name)}
              </Link>
            ))}
          </nav>

          {/* Mobile-only: menu + dark toggle, spaced */}
          <div className="flex lg:hidden items-center space-x-4">
            {/* menu button */}
            <button
              onClick={toggleMenu}
              className="text-green-600 hover:text-green-900 dark:text-green-300 dark:hover:text-green-100"
              aria-label="Toggle menu"
            >
              {isMenuOpen ? <HiX size={24} /> : <HiMenu size={24} />}
            </button>

            {/* dark/light toggle */}
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
          </div>
        </div>

        {/* Desktop-only right group */}
        <div className="hidden lg:flex items-center space-x-4">
          {/* dark/light toggle */}
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

          {/* language selector */}
          <LanguageDropdown />

          {/* auth buttons */}
          {user ? (
            <>
              <span className="text-sm text-gray-700 dark:text-gray-300">
                Hi, {user.name}
              </span>
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
                className="mr-2 bg-white text-green-600 border border-green-600 px-4 py-2 rounded-md text-sm font-medium dark:bg-gray-900 dark:hover:bg-green-600 dark:hover:text-gray-900 hover:bg-green-700 hover:text-white"
              >
                {t("Sign Up")}
              </Link>
              <Link
                href="/login"
                className="mr-2 bg-white text-green-600 border border-green-600 px-4 py-2 rounded-md text-sm font-medium dark:bg-gray-900 dark:hover:bg-green-600 dark:hover:text-gray-900 hover:bg-green-700 hover:text-white"
              >
                {t("Log In")}
              </Link>
            </>
          )}
        </div>
      </div>

      {/* Mobile nav drawer */}
      {isMenuOpen && (
        <nav className="lg:hidden bg-white dark:bg-gray-900 px-4 pb-4 space-y-1">
          {navItems.map((item) => (
            <Link
              key={item.name}
              href={item.href}
              className="block text-green-600 hover:text-green-900 dark:text-green-300 dark:hover:text-green-100 px-3 py-2 text-base font-medium"
            >
              {t(item.name)}
            </Link>
          ))}

          {/* mobile auth */}
          {user ? (
            <>
              <span className="block text-green-800 dark:text-green-300 px-3 py-2 text-base font-medium">
                Hi, {user.name}
              </span>
>>>>>>> Stashed changes
              <button
                onClick={toggleMenu}
                className="text-green-600 hover:text-green-900 focus:outline-none focus:text-green-900"
              >
                {isMenuOpen ? <HiX className="h-6 w-6" /> : <HiMenu className="h-6 w-6" />}
              </button>
            </div>
            {/* Menu Items */}
            <nav className={`ml-10 space-x-4 hidden lg:flex ${isMenuOpen ? 'block' : 'hidden'} lg:block`}>
              {navItems.map((item) => (
                <Link
                  key={item.name}
                  href={item.link}
                  className="text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-sm font-medium"
                >
                  {t(item.name)}
                </Link>
              ))}
            </nav>
          </div>
          <div className="flex items-center">
            <LanguageDropdown />
            <div className='hidden lg:flex'>
            <Link
              href="/signup"
              className="bg-green-600 text-white hover:bg-green-700 px-4 py-2 rounded-md text-sm font-medium"
            >
              {t('Sign Up')}
            </Link>
            <Link
              href="/login"
              className="ml-4 bg-white text-green-600 hover:bg-gray-50 border border-green-600 px-4 py-2 rounded-md text-sm font-medium"
            >
              {t('Log In')}
            </Link>
            </div>
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
                  {t(item.name)}
                </Link>
              ))}
              {/* Add Sign Up and Log In buttons to mobile menu */}
              <Link
                href="/signup"
                className="block text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-base font-medium"
              >
                {t('Sign Up')}
              </Link>
              <Link
                href="/login"
                className="block text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-base font-medium"
              >
                {t('Log In')}
              </Link>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;
