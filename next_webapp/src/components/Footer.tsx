import React from 'react';
import { Link } from "@/i18n-navigation";
import { useTranslations } from "next-intl";
import { FaFacebook, FaTwitter, FaLinkedin } from 'react-icons/fa';

const Footer = () => {
  const t = useTranslations("common");

  return (
    <footer className="bg-green-500 text-white py-8 mt-16 md:mt-10">
      <div className="container mx-auto px-4 md:px-2 lg:px-8">
        {/* Flexbox container for layout */}
        <div className="flex flex-col md:flex-row items-center justify-between space-y-8 md:space-y-0 md:space-x-8">
          
          {/* Logo and Navigation */}
          <div className="flex flex-col md:flex-row items-center md:space-x-6 space-y-8 md:space-y-0">
            {/* Logo */}
            <img src="/img/new-logo-white.png" alt="Hameleon logo" className="h-16 lg:h-24 w-auto" />

            {/* Navigation Links */}
            <nav className="mt-4 md:mt-0">
              <ul className="flex flex-col md:flex-row items-center space-y-4 md:space-y-0 md:space-x-6 text-base lg:text-lg">
                <li>
                  <Link href="/licensing" className="hover:underline">
                    {t("Licensing")}
                  </Link>
                </li>
                <li>
                  <Link href="/privacypolicy" className="hover:underline">
                    {t("Privacy Policy")}
                  </Link>
                </li>
                <li>
                  <Link href="/contact" className="hover:underline">
                    {t("Contact Us")}
                  </Link>
                </li>
              </ul>
            </nav>
          </div>

          {/* Social Icons and External Link */}
          <div className="flex flex-col md:flex-row items-center space-y-8 md:space-y-0 md:space-x-6">
            {/* Social Media Icons */}
            <div className="flex space-x-4">
              <a href="#" aria-label="Facebook" className="hover:text-gray-300">
                <FaFacebook size={24} />
              </a>
              <a href="#" aria-label="Twitter" className="hover:text-gray-300">
                <FaTwitter size={24} />
              </a>
              <a href="#" aria-label="LinkedIn" className="hover:text-gray-300">
                <FaLinkedin size={24} />
              </a>
            </div>

            {/* External Link */}
            <a href="https://data.melbourne.vic.gov.au/pages/home/" className="flex items-center hover:underline">
              Melbourne Open Data
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" viewBox="0 0 20 20" fill="currentColor">
                <path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z" />
                <path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z" />
              </svg>
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;