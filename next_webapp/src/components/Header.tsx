"use client"
import React, { useState } from "react";
import LanguageDropdown from "../components/LanguageDropdown";
import { Link } from "@/i18n-navigation";
import { useTranslations } from "next-intl";
import { CiMenuBurger  } from "react-icons/ci";
import { HiMiniXMark } from "react-icons/hi2";

const Header = () => {
  const t = useTranslations("common");
  let[isOpen, setisOpen] = useState(false);
 

  return (
    <header>
      <div>
        <nav className="border-1 border white-200 bg-white text-white">
          <div className=" md:flex items-center justify-between px-20 py-4">
            <div className="flex items-center">
              <a href="/">
                <img src="/img/new-logo-green.png" className="h-20 w-20 absolute left-1 top-4"   alt="MOP logo" />
              </a>
              <div onClick={()=>setisOpen(!isOpen)}className="w-10 h-7 absolute right-8 top-6 cursor-pointer md:hidden">
                {
                  isOpen ? <HiMiniXMark className="size-8"/>: <CiMenuBurger className="size-6"/>  
                }
              </div>
              <ul className={`md:flex md:items-right-10 md:space-x-6 ${
                isOpen ? "block" : "hidden"
              } md:block`}>
                <li className="">
                  <a
                    href="/"
                    className="rounded-3xl hover:bg-[#287405] block font-bold py-4 px-5 text-green-500  ml-3 text-lg"
                    aria-current="page"
                  >
                    {t("Home")}
                  </a>
                </li>
                <li className=" md:inline-block">
                  <Link
                    href="/about"
                    className="rounded-3xl hover:bg-[#287405] block font-bold py-4 px-5 text-green-500  text-lg"
                  >
                    {t("About Us")}
                  </Link>
                </li>
                <li className=" md:inline-block">
                  <Link
                    href="/UseCases"
                    className="rounded-3xl hover:bg-[#287405] block font-bold py-4 px-5 text-green-500  text-lg"
                  >
                    {t("Use Cases")}
                  </Link>
                </li>
                <li className=" md:inline-block">
                  <Link
                    href="/statistics"
                    className="rounded-3xl hover:bg-[#287405] block font-bold py-4 px-5 text-green-500  text-lg"
                  >
                    {t("Statistics")}
                  </Link>
                </li>
                <li className=" md:inline-block">
                  <Link
                    href="/upload"
                    className="rounded-3xl hover:bg-[#287405] block font-bold py-4 px-5 text-green-500  text-lg"
                  >
                    {t("Upload")}
                  </Link>
                </li>
              </ul>
            </div>
            <ul className={`md:flex justify-between md:pl-9 absolute bg-white-400  md:pr-[15rem] ml-0 py-5 md:w-auto w-full md:static md:z-auto z-[1] ${isOpen ? 'top-[21rem]' : 'top-[-430px]'}` }>
              <li className="">
              <div className=" " x-data="{ open: false }">
             
                <LanguageDropdown />
              </div>
              </li>
              <li className=" mt-5 sm:mt-3">
              <Link
                href="/signup"
                className="border   mr-3 font-serif py-3 px-6 mx-3 text-white rounded-md  text-lg bg-green-400 hover:bg-blue-600"
              >
                {t("Sign Up")}
              </Link>
              </li>
              <li className="mt-8 sm:mt-3">
              <Link
                href="/login"
                className="border border-solid border-green-500 bg-white text-green-500 font-serif py-3 px-6 mx-3 rounded-md  text-lg"
              >
                {t("Log In")}
              </Link>
              </li>
            </ul>
          </div>
        </nav>
      </div>
    </header>
  );
};

export default Header;
