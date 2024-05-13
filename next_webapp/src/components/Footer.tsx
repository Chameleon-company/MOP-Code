// Footer.js
import { Link } from "@/i18n-navigation";
import React from "react";
import { useTranslations } from "next-intl";

const Footer = () => {
  const t = useTranslations("common");

  return (
    <footer class="bg-white text-black justify-center mt-10">
      <hr class=" h-1 border-1 text-gray-500 bg-gray-500 mx-8"></hr>
      <div class="w-full px-32 py-7 flex items-center">
        <div class="flex-shrink-0 mr-4">
          <img src="/img/new-logo-green.png" alt="logo" class="w-20" />
        </div>
        <div class="flex-grow text-center">
          <ul class="flex justify-center gap-20">
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
        <div class="flex-shrink-0 ml-4 flex gap-2">
          <a href="https://data.melbourne.vic.gov.au/pages/home/">
            {t("Melbourne Open Data")}
          </a>
          <img
            src="https://www.svgrepo.com/show/510970/external-link.svg"
            alt="link icon"
            class="w-5"
          />
        </div>
      </div>
    </footer>
  );
};

export default Footer;
