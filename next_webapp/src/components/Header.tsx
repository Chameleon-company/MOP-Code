import React from "react";
import LanguageDropdown from "../components/LanguageDropdown";
import { useTranslations } from "next-intl";

const Header = () => {
  const t = useTranslations("common");
  return (
    <header>
      <div>
        <nav className="border-gray-200 bg-green-400 text-white">
          <div className="max-w-screen-xl flex items-center justify-between mx-auto p-4">
            <div className="flex items-center space-x-3">
              <a href="/">
                <img src="/img/new-logo-white.png" className="h-20" alt="MOP logo" />
              </a>
              <ul className="flex justify-evenly ml-8">
                <li className="inline-block">
                  <a
                    href="/"
                    className="rounded-3xl hover:bg-[#287405] block font-serif py-4 px-5 text-white  ml-3 text-lg"
                    aria-current="page"
                  >
                    {t("Home")}
                  </a>
                </li>
                <li className="inline-block">
                  <a
                    href="/en/about"
                    className="rounded-3xl hover:bg-[#287405] block font-serif py-4 px-5 text-white  text-lg"
                  >
                    {t("About Us")}
                  </a>
                </li>
                <li className="inline-block">
                  <a
                    href="/en/UseCases"
                    className="rounded-3xl hover:bg-[#287405] block font-serif py-4 px-5 text-white  text-lg"
                  >
                    {t("Use Cases")}
                  </a>
                </li>
                <li className="inline-block">
                  <a
                    href="/en/statistics"
                    className="rounded-3xl hover:bg-[#287405] block font-serif py-4 px-5 text-white  text-lg"
                  >
                    {t("Statistics")}
                  </a>
                </li>
                <li className="inline-block">
                  <a
                    href="/en/upload"
                    className="rounded-3xl hover:bg-[#287405] block font-serif py-4 px-5 text-white  text-lg"
                  >
                    {t("Upload")}
                  </a>
                </li>
              </ul>
            </div>
            <div className="flex items-center justify-end">
              <div className="relative" x-data="{ open: false }">
                <LanguageDropdown />
              </div>
              <a
                href="/en/signup"
                className="border-[1px] border-solid border-white mr-3 font-serif py-3 px-6 mx-3 text-white rounded-full  text-lg"
              >
                {t("Sign Up")}
              </a>
              <a
                href="/en/login"
                className="border-[1px] border-solid border-white bg-white text-[#09bd09] font-serif py-3 px-6 mx-3 rounded-full  text-lg"
              >
                {t("Log In")}
              </a>
            </div>
          </div>
        </nav>
      </div>
    </header>
  );
};

export default Header;
