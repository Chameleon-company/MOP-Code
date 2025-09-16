"use client";
import React, { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { Link } from "@/i18n-navigation";
import LanguageDropdown from "./LanguageDropdown";
import { HiMenu, HiX, HiMoon, HiSun } from "react-icons/hi";
import { useTheme } from "../hooks/useTheme";
import Calendar from "react-calendar"; // Importing the calendar component
import "react-calendar/dist/Calendar.css"; // Import the calendar styles

const Header = () => {
  const t = useTranslations("common");
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [events, setEvents] = useState([]);
  const [isClient, setIsClient] = useState(false); // Client-side check
  const { theme, toggleTheme } = useTheme();

  // To fix hydration issue, check if we're on the client side
  useEffect(() => {
    setIsClient(true); // Set to true once the component is mounted on the client
  }, []);

  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);

  const navItems = [
    { name: "Home", link: "/" },
    { name: "About Us", link: "/about" },
    { name: "Use Cases", link: "/usecases" },
    { name: "Statistics", link: "/statistics" },
    { name: "Upload", link: "/upload" },
  ];

  // Mock Events
  const mockEvents = {
    "2025-08-03": [
      {
        id: 1,
        title: "Local Music Fest",
        time: "3:00 PM",
        description: "Live bands at Federation Square.",
      },
      {
        id: 2,
        title: "Art Walk",
        time: "6:00 PM",
        description: "Walking tour of city murals.",
      },
    ],
    "2025-08-05": [
      {
        id: 3,
        title: "Sustainability Forum",
        time: "10:00 AM",
        description: "Panel talk on smart cities.",
      },
    ],
  };

  useEffect(() => {
    const key = selectedDate.toISOString().split("T")[0];
    setEvents(mockEvents[key] || []);
  }, [selectedDate]);

  return (
    <header className="bg-white dark:bg-black shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <Link href="/" className="flex-shrink-0">
              <img
                className="h-20 w-auto"
                src="/img/new-logo-green.png"
                alt="Logo"
              />
            </Link>

            <div className="flex lg:hidden ml-auto">
              <button
                onClick={toggleMenu}
                className="text-green-600 hover:text-green-900"
              >
                {isMenuOpen ? (
                  <HiX className="h-6 w-6" />
                ) : (
                  <HiMenu className="h-6 w-6" />
                )}
              </button>
            </div>

            <nav
              className={`ml-10 space-x-4 hidden lg:flex ${
                isMenuOpen ? "block" : "hidden"
              } lg:block`}
            >
              {navItems.map((item) => (
                <Link
                  key={item.name}
                  href={item.link}
                  className="text-black dark:text-gray-200 hover:text-green-900 dark:hover:text-green-300 px-3 py-2 rounded-md text-sm font-medium"
                >
                  {t(item.name)}
                </Link>
              ))}
              {/* Sign Up and Log In in the navigation bar */}
              <Link
                href="/signup"
                className="text-black dark:text-gray-200 hover:text-green-900 dark:hover:text-green-300 px-3 py-2 rounded-md text-sm font-medium"
              >
                {t("Sign Up")}
              </Link>
              <Link
                href="/login"
                className="text-black dark:text-gray-200 hover:text-green-900 dark:hover:text-green-300 px-3 py-2 rounded-md text-sm font-medium"
              >
                {t("Log In")}
              </Link>
            </nav>
          </div>

          <div className="flex items-center">
            <button
              onClick={toggleTheme}
              aria-label="Toggle Dark Mode"
              className="p-1 rounded focus:outline-none"
            >
              {theme === "dark" ? (
                <HiSun className="mr-4 h-5 w-5 text-white" />
              ) : (
                <HiMoon className="mr-4 h-5 w-5 text-black" />
              )}
            </button>

            <LanguageDropdown />
          </div>
        </div>
      </div>

      {/* Event Calendar */}
      <div className="calendar-container mt-6 bg-white dark:bg-gray-800 p-4 rounded-lg shadow-md flex">
        {/* Calendar Component */}
        <div className="calendar w-full lg:w-2/3">
          <h3 className="text-xl font-bold mb-2 text-black dark:text-white">
            Upcoming Events
          </h3>
          <Calendar onChange={setSelectedDate} value={selectedDate} />
        </div>

        {/* Event Details */}
        {isClient && ( // Ensure event display happens only on the client side
          <div className="events-list w-full lg:w-1/3 mt-4 lg:mt-0 lg:pl-8">
            <h4 className="text-lg font-semibold mb-2 text-black dark:text-white">
              Events on {selectedDate.toLocaleDateString()}
            </h4>
            {events.length > 0 ? (
              events.map((event) => (
                <div
                  key={event.id}
                  className="event-item mb-4 p-4 bg-gray-100 dark:bg-gray-700 rounded-lg shadow-md"
                >
                  <p className="text-sm font-semibold text-black dark:text-white">
                    {event.title}
                  </p>
                  <p className="text-xs text-gray-800 dark:text-gray-300">
                    {event.time}
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {event.description}
                  </p>
                </div>
              ))
            ) : (
              <p className="text-gray-600 dark:text-gray-400">
                No events for this date.
              </p>
            )}
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;
