'use client';
import React, { useState } from 'react';
import { useTranslations } from 'next-intl';
import { Link } from '@/i18n-navigation';
import LanguageDropdown from './LanguageDropdown';
import { HiMenu, HiX } from 'react-icons/hi';
import { useDarkMode } from '@/context/DarkModeContext'; // Import the custom dark mode hook

const Header = () => {
  const t = useTranslations('common');
  const { darkMode } = useDarkMode(); // Access the dark mode state
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const navItems = [
    { name: 'Home', link: '/' },
    { name: 'About Us', link: '/about' },
    { name: 'Use Cases', link: '/UseCases' },
    { name: 'Statistics', link: '/statistics' },
    { name: 'Upload', link: '/upload' }
  ];

  return (
    <header
      className={`${
        darkMode ? 'bg-[#1D1919] text-white' : 'bg-white text-gray-900'
      } shadow-sm transition-colors duration-300`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/" className="flex-shrink-0">
            <img
              className="h-20 w-auto"
              src={darkMode ? '/img/new-logo-white.png' : '/img/new-logo-green.png'}
              alt="Logo"
            />
          </Link>

          {/* Mobile Menu Icon */}
          <div className="flex lg:hidden ml-auto">
            <button
              onClick={toggleMenu}
              className={`${
                darkMode ? 'text-white' : 'text-green-600'
              } focus:outline-none`}
            >
              {isMenuOpen ? <HiX className="h-6 w-6" /> : <HiMenu className="h-6 w-6" />}
            </button>
          </div>

          <nav
            className={`ml-10 space-x-4 hidden lg:flex ${
              isMenuOpen ? 'block' : 'hidden'
            } lg:block`}
          >
            {navItems.map((item) => (
              <Link
                key={item.name}
                href={item.link}
                className={`${
                  darkMode
                    ? 'text-white hover:text-green-400'
                    : 'text-green-600 hover:text-green-900'
                } px-3 py-2 rounded-md text-sm font-medium`}
              >
                {t(item.name)}
              </Link>
            ))}
          </nav>

          {/* Buttons and Language Dropdown */}
          <div className="flex items-center">
            <LanguageDropdown />
            <div className="hidden lg:flex">
              <Link
                href="/signup"
                  className={`ml-4 ${
                    darkMode
                      ? 'bg-white text-green-700 border border-green-700 hover:bg-green-700 hover:text-white'
                      : 'bg-white text-green-700 border border-green-600 hover:bg-green-600 hover:text-white'
                  } px-4 py-2 rounded-md text-sm font-medium transition-colors duration-300`}
                >
                {t('Sign Up')}
              </Link>
              <Link
                href="/login"
                className={`ml-4 ${
                  darkMode
                    ? 'bg-white text-green-700 border border-green-700 hover:bg-green-700 hover:text-white'
                    : 'bg-white text-green-700 border border-green-600 hover:bg-green-600 hover:text-white'
                } px-4 py-2 rounded-md text-sm font-medium transition-colors duration-300`}
              >
                {t("Log In")}
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
                  className={`block ${
                    darkMode ? 'text-white hover:text-green-400' : 'text-green-600 hover:text-green-900'
                  } px-3 py-2 rounded-md text-base font-medium`}
                >
                  {t(item.name)}
                </Link>
              ))}
              <Link
                href="/signup"
                className={`block ${
                  darkMode ? 'text-white hover:text-green-400' : 'text-green-600 hover:text-green-900'
                } px-3 py-2 rounded-md text-base font-medium`}
              >
                {t('Sign Up')}
              </Link>
              <Link
                href="/login"
                className={`block ${
                  darkMode ? 'text-white hover:text-green-400' : 'text-green-600 hover:text-green-900'
                } px-3 py-2 rounded-md text-base font-medium`}
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
