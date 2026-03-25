'use client'
import React from 'react';
import { Link } from "@/i18n-navigation";
import { useTranslations } from "next-intl";
import { FaFacebook, FaLinkedin } from 'react-icons/fa';
import { FaSquareXTwitter } from "react-icons/fa6";

const Footer = () => {
  const t = useTranslations("common");

  return (
    // Main footer container with background color and padding
    <footer className="bg-green-500 dark:bg-green-800 text-white pt-12 pb-6">
      <div className="container mx-auto px-6 lg:px-12">

        {/* Grid layout creates 3 columns (responsive) and vertical divider lines */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-10 md:divide-x md:divide-green-300 md:divide-opacity-80 items-start pt-0">

          {/* Flex column with spacing adjustments for logo and text */}
          <div className="flex flex-col items-center md:items-start md:pr-10">
            <img
              src="/img/new-logo-white.png"
              alt="Melbourne Open Playground logo"
              className="h-24 lg:h-32 w-auto -mt-6" // Increased size + adjusted position
            />
            <p className="text-white text-base text-center md:text-left leading-relaxed -mt-3">
              Exploring Melbourne&#39;s open data
              <br />
              for a smarter city.
            </p>
          </div>

          {/* Vertical navigation links with spacing and hover animation */}
          <div className="flex flex-col items-center md:items-start space-y-3 md:px-10">
            <h3 className="text-lg font-semibold tracking-wide uppercase text-white mb-1">
              Quick Links
            </h3>

            {/* Hover effect: slight movement + color change */}
            <Link
              href="/licensing"
              className="text-base text-white hover:text-gray-200 hover:translate-x-1 transition-all duration-200"
            >
              {t("Licensing")}
            </Link>

            <Link
              href="/privacypolicy"
              className="text-base text-white hover:text-gray-200 hover:translate-x-1 transition-all duration-200"
            >
              {t("Privacy Policy")}
            </Link>

            <Link
              href="/contact"
              className="text-base text-white hover:text-gray-200 hover:translate-x-1 transition-all duration-200"
            >
              {t("Contact Us")}
            </Link>
          </div>

          {/* External link + social media icons */}
          <div className="flex flex-col items-center md:items-start space-y-4 md:pl-10">
            <h3 className="text-lg font-semibold tracking-wide uppercase text-white mb-1">
              Connect
            </h3>

            {/* External link with hover animation */}
            <a
              href="https://data.melbourne.vic.gov.au/pages/home/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center text-base text-white hover:text-gray-200 transition-colors duration-200 group"
            >
              Melbourne Open Data

              {/* Small icon animation on hover */}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-4 w-4 ml-1 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform duration-200"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z" />
                <path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z" />
              </svg>
            </a>

            <div>
              <p className="text-sm text-white uppercase tracking-widest mb-3">
                Follow us
              </p>

              {/* Icons arranged horizontally with spacing */}
              <div className="flex space-x-4">
                
                {/* Hover effects: scale + color inversion */}
                <a
                  href="#"
                  aria-label="Facebook"
                  className="p-2 rounded-full bg-green-600 hover:bg-white hover:text-green-600 text-white transition-all duration-200 hover:scale-110"
                >
                  <FaFacebook size={22} />
                </a>
                
                <a
                  href="#"
                  aria-label="Twitter / X"
                  className="p-2 rounded-full bg-green-600 hover:bg-white hover:text-green-600 text-white transition-all duration-200 hover:scale-110"
                >
                  <FaSquareXTwitter size={22} />
                </a>
                
                <a
                  href="#"
                  aria-label="LinkedIn"
                  className="p-2 rounded-full bg-green-600 hover:bg-white hover:text-green-600 text-white transition-all duration-200 hover:scale-110"
                >
                  <FaLinkedin size={22} />
                </a>

              </div>
            </div>
          </div>

        </div>

        {/* Horizontal divider line for separation */}
        <div className="border-t-2 border-green-300 border-opacity-80 mt-10 pt-5">
          <p className="text-center text-sm text-white">
            {`© ${new Date().getFullYear()} Melbourne Open Playground · City of Melbourne Open Data · Deakin University`}
          </p>
        </div>

      </div>
    </footer>
  );
};

export default Footer;