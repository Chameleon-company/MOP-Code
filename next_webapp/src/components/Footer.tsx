"use client";
import { Link } from "@/i18n-navigation";
import React from "react";
import { useTranslations } from "next-intl";

const Footer = () => {
  const t = useTranslations("common");

  return (
    <footer className="bg-green-600 text-black justify-center mt-10">
      <hr className="h-1 border-0 text-gray-500 bg-white-500 mx-8" />
      <div className="w-full px-32 py-7 flex items-center">
        <div className="absolute left-0.5 pl-0.5">
          <img src="/img/new-logo-white.png" alt="logo" className="w-24" />
        </div>
        <div className="flex-grow text-center">
          <ul className="flex justify-left gap-16 text-lg text-white">
            <Link href="/Licensing">
              <li>{t("Licensing")}</li>
            </Link>
            <Link href="/Privacy Policy">
              <li>{t("Privacy Policy")}</li>
            </Link>
            <Link href="/contact">
              <li>{t("Contact Us")}</li>
            </Link>
          </ul>
        </div>
        <div className="flex-shrink-0 ml-4 flex gap-4 items-center">
          <a
            href="https://data.melbourne.vic.gov.au/pages/home/"
            className="flex items-center"
          >
            {t("Melbourne Open Data")}
            <img
              src="https://www.svgrepo.com/show/510970/external-link.svg"
              alt="link icon"
              className="w-5 ml-1"
            />
          </a>
          {/* Social Media Links */}
          <a
            href="https://www.facebook.com"
            target="_blank"
            rel="noopener noreferrer"
          >
            <img
              src="https://www.svgrepo.com/svg/303163/facebook-4-logo.svg"
              alt="Facebook"
              className="w-6 h-6"
            />
          </a>
          <a href="https://x.com" target="_blank" rel="noopener noreferrer">
            <img
              src="https://www.svgrepo.com/svg/137277/twitter.svg"
              alt="X"
              className="w-6 h-6"
            />
          </a>
          <a
            href="https://www.linkedin.com"
            target="_blank"
            rel="noopener noreferrer"
          >
            <img
              src="https://www.svgrepo.com/show/157006/linkedin.svg"
              alt="LinkedIn"
              className="w-6 h-6"
            />
          </a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
