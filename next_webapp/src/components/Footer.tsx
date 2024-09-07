import React from 'react';
import { Link } from "@/i18n-navigation";
import { useTranslations } from "next-intl";
import { FaFacebook, FaTwitter, FaLinkedin } from 'react-icons/fa';

const Footer = () => {
  const t = useTranslations("common");

  return (
    <footer className="bg-green-500 text-white py-4 mt-20">
      <div className="container mx-auto px-4">
        <div className="flex flex-col lg:flex-row items-center justify-between space-y-4 lg:space-y-0">
          <div className="flex flex-col lg:flex-row items-center lg:space-x-8 space-y-4 lg:space-y-0">
            <img src="/img/new-logo-white.png" alt="Hameleon logo" className="h-20 lg:h-28" />
            <nav className="mt-4 lg:mt-0">
              <ul className="flex flex-col lg:flex-row items-center space-y-2 lg:space-y-0 lg:space-x-6 text-lg lg:text-xl">
                <li><Link href="/licensing" className="hover:underline">Licensing</Link></li>
                <li><Link href="/privacypolicy" className="hover:underline">Privacy Policy</Link></li>
                <li><Link href="/contact" className="hover:underline">Contact Us</Link></li>
              </ul>
            </nav>
          </div>
          
          <div className="flex flex-col items-center lg:flex-row lg:items-center space-y-4 lg:space-y-0 lg:space-x-6">
            <div className="flex space-x-4">
              <a href="#" aria-label="Facebook"><FaFacebook size={20} /></a>
              <a href="#" aria-label="Twitter"><FaTwitter size={20} /></a>
              <a href="#" aria-label="LinkedIn"><FaLinkedin size={20} /></a>
            </div>
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