"use client";
import React, { useState } from "react";
import LanguageDropdown from "../components/LanguageDropdown";
import "../../public/styles/header.css";
import { CiMenuBurger  } from "react-icons/ci";
import { HiMiniXMark } from "react-icons/hi2";

 // Import the CSS file

const Header = () => {
  let[isOpen, setisOpen] = useState(false);
  return (
    <header>
      <div>
        <nav className="border-gray-200 bg-green-400 text-white py-3">
          <div className=" md:flex items-center justify-between ">
            <div className="md:flex items-center  ">
              <a href="/">
                <img src="/img/image.png" className="h-8" alt="MOP logo" />
              </a>
              <div onClick={()=>setisOpen(!isOpen)} className="w-10 h-7 absolute right-8 top-6 cursor-pointer md:hidden">
                {
                  isOpen ?  <CiMenuBurger className="size-6"/> :   <HiMiniXMark className="size-8"/>
                }
              </div>
              <ul className={`md:flex justify-between md:pl-9 bg-green-400  md:w-auto w-full absolute md:static md:z-auto  z-[1] ${isOpen ? 'top-12' : 'top-[-430px]'}` }>
                <li className="">
                  <a
                    href="/"
                    className="nav-link block font-serif py-4 px-5 text-white rounded  ml-3 text-lg"
                    aria-current="page"
                  >
                    Home
                  </a>
                </li>
                <li className="">
                  <a
                    href="/about"
                    className="nav-link block font-serif py-4 px-5 text-white rounded  text-lg"
                  >
                    About Us
                  </a>
                </li>
                <li className="">
                  <a
                    href="/UseCases"
                    className="nav-link block font-serif py-4 px-5 text-white rounded  text-lg"
                  >
                    Use Cases
                  </a>
                </li>
                <li className="">
                  <a
                    href="/statistics"
                    className="nav-link block font-serif py-4 px-5 text-white rounded  text-lg"
                  >
                    Statistics
                  </a>
                </li>
                <li className="">
                  <a
                    href="/upload"
                    className="nav-link block font-serif py-4 px-5 text-white rounded  text-lg"
                  >
                    Upload
                  </a>
                </li>
              </ul>
            </div>
            <ul className={`md:flex justify-between md:pl-9 absolute bg-green-400   ml-0 py-5 md:w-auto w-full md:static md:z-auto z-[1] ${isOpen ? 'top-[21rem]' : 'top-[-430px]'}` }>
              <li className="">
              <div className=" " x-data="{ open: false }">
                <LanguageDropdown />
              </div>
              </li>
              <li className=" mt-5 sm:mt-3">
              <a
                href="/signup"
                className="signup-btn font-serif py-3 px-6 mx-3  text-white rounded-full  text-lg"
              >
                Sign Up
              </a>
              </li>
              <li className="mt-8 sm:mt-3">
              <a
                href="/login"
                className="login-btn font-serif py-3 px-6 mx-3  text-white rounded-full  text-lg"
              >
                Log In
              </a>
              </li>
            </ul>
          </div>
        </nav>
      </div>
    </header>
  );
};

export default Header;
