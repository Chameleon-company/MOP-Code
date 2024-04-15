import React from "react";
import Link from "next/link";

const Header = () => {
  return (
    <header>
      <div>
        <nav className="border-gray-200 bg-green-400 text-white">
          <div className="max-w-screen-xl flex items-center justify-between mx-auto p-4">
            <div className="flex items-center space-x-3">
              <a href="/">
                <img src="/img/image.png" className="h-8" alt="MOP logo" />
              </a>
              <ul className="flex justify-evenly ml-8">
                <li className="inline-block">
                  <Link
                    href="/"
                    className="block font-serif py-4 px-5 text-white rounded hover:bg-blue-400 ml-3 text-lg"
                    aria-current="page"
                  >
                    Home
                  </Link>
                </li>
                <li className="inline-block">
                  <Link
                    href="/about"
                    className="block font-serif py-4 px-5 text-white rounded hover:bg-blue-400 text-lg"
                  >
                    about us
                  </Link>
                </li>
                <li className="inline-block">
                  <Link
                    href="/casestudies"
                    className="block font-serif py-4 px-5 text-white rounded hover:bg-blue-400 text-lg"
                  >
                    Case Studies
                  </Link>
                </li>
                <li className="inline-block">
                  <Link
                    href="/contact"
                    className="block font-serif py-4 px-5 text-white rounded hover:bg-blue-400 text-lg"
                  >
                    Contact us
                  </Link>
                </li>
              </ul>
            </div>
          </div>
        </nav>
      </div>
    </header>
  );
};

export default Header;

