// Footer.js
"use client";
import { Link } from "@/i18n-navigation";
import React from "react";
import { useTranslations } from "next-intl";
import { useState } from "react";

const Footer = () => {
  const t = useTranslations("common");
 //dark theme
 const [dark_value,setdarkvalue] = useState(false);
  
 const handleValueChange = (newValue: boolean | ((prevState: boolean) => boolean))=>{
   setdarkvalue(newValue);
 }
  return (
    <div className= {`${dark_value && "dark"}`}>
    <footer className="bg-white text-black justify-center mt-10 dark:bg-sc_bg_dark dark:text-white">
      <hr className=" h-1 border-1 text-gray-500 bg-gray-500 mx-8"></hr>
      <div className="w-full px-32 py-7 flex items-center">
        <div className="flex-shrink-0 mr-4">
          <img src="/img/new-logo-green.png" alt="logo" className="w-20" />
        </div>
        <div className="flex-grow text-center">
          <ul className="flex justify-center gap-20">
            <Link href="/privacypolicy">
              <li>{t("Privacy Policy")}</li>
            </Link>
            <Link href="/licensing">
              <li>{t("Licensing")}</li>
            </Link>
            <Link href="/contact">
              <li>{t("Contact Us")}</li>
            </Link>
          </ul>
        </div>
        <div className="flex-shrink-0 ml-4 flex gap-2">
          <a href="https://data.melbourne.vic.gov.au/pages/home/">
            {t("Melbourne Open Data")}
          </a>
          <img
            src="https://www.svgrepo.com/show/510970/external-link.svg"
            alt="link icon"
            className="w-5"
          />
        </div>
      </div>
    </footer>
    </div>
  );
};

export default Footer;
