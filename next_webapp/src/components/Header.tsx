"use client"
import React, { useState } from "react";
import LanguageDropdown from "../components/LanguageDropdown";
import { Link } from "@/i18n-navigation";
import { useTranslations } from "next-intl";

const Header = () => {
  const t = useTranslations("common");

  return (
    <header>
      <div>
        <nav className="border-gray-200 bg-green-400 text-white py-3">
          <div className=" md:flex items-center justify-between ">
            <div className="md:flex items-center  ">
              <a href="/">
                <img src="/img/new-logo-white.png" className="h-20" alt="MOP logo" />
              </a>
              <div onClick={()=>handleChange()} className="w-10 h-7 absolute right-8 top-6 cursor-pointer md:hidden">
                {
                  isOpen ? <HiMiniXMark className="size-8"/>: <CiMenuBurger className="size-6"/>  
                }
              </div>
              <ul className={`md:flex justify-between md:pl-9 bg-green-400  md:w-auto w-full absolute md:static md:z-auto  z-[1] ${isOpen ? 'top-12' : 'top-[-430px]'}` }>
                <li className="">
                  <a
                    href="/"
                    className="rounded-3xl hover:bg-[#287405] block font-serif py-4 px-5 text-white  ml-3 text-lg"
                    aria-current="page"
                  >
                    {t("Home")}
                  </a>
                </li>
                <li className="inline-block">
                  <Link
                    href="/about"
                    className="rounded-3xl hover:bg-[#287405] block font-serif py-4 px-5 text-white  text-lg"
                  >
                    {t("About Us")}
                  </Link>
                </li>
                <li className="inline-block">
                  <Link
                    href="/UseCases"
                    className="rounded-3xl hover:bg-[#287405] block font-serif py-4 px-5 text-white  text-lg"
                  >
                    {t("Use Cases")}
                  </Link>
                </li>
                <li className="inline-block">
                  <Link
                    href="/statistics"
                    className="rounded-3xl hover:bg-[#287405] block font-serif py-4 px-5 text-white  text-lg"
                  >
                    {t("Statistics")}
                  </Link>
                </li>
                <li className="inline-block">
                  <Link
                    href="/upload"
                    className="rounded-3xl hover:bg-[#287405] block font-serif py-4 px-5 text-white  text-lg"
                  >
                    {t("Upload")}
                  </Link>
                </li>
              </ul>
            </div>
            <ul className={`md:flex justify-between md:pl-9 absolute bg-green-400   ml-0 py-5 md:w-auto w-full md:static md:z-auto z-[1] ${isOpen ? 'top-[21rem]' : 'top-[-430px]'}` }>
              <li className="">
              <div className=" " x-data="{ open: false }">
                <LanguageDropdown />
              </div>
              <Link
                href="/signup"
                className="border-[1px] border-solid border-white mr-3 font-serif py-3 px-6 mx-3 text-white rounded-full  text-lg"
              >
                {t("Sign Up")}
              </Link>
              <Link
                href="/login"
                className="border-[1px] border-solid border-white bg-white text-[#09bd09] font-serif py-3 px-6 mx-3 rounded-full  text-lg"
              >
                {t("Log In")}
              </Link>
            </div>
          </div>
        </nav>
      </div>
    </header>
  );
};

export default Header;
