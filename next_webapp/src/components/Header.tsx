import React from "react";
import "../../public/styles/header.css";// Import the CSS file

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
                  <a
                    href="/"
                    className="nav-link block font-serif py-4 px-5 text-white rounded  ml-3 text-lg" 
                    aria-current="page"
                  >
                    Home
                  </a>
                </li>
                <li className="inline-block">
                  <a
                    href="/about"
                    className="nav-link block font-serif py-4 px-5 text-white rounded  text-lg"
                  >
                    About Us
                  </a>
                </li>
                <li className="inline-block">
                  <a
                    href="/UseCases"
                    className="nav-link block font-serif py-4 px-5 text-white rounded  text-lg"
                  >
                    Use Cases
                  </a>
                </li>
                <li className="inline-block">
                  <a
                    href="/statistics"
                    className="nav-link block font-serif py-4 px-5 text-white rounded  text-lg"
                  >
                    Statistics
                  </a>
                </li>
                <li className="inline-block">
                  <a
                    href="/upload"
                    className="nav-link block font-serif py-4 px-5 text-white rounded  text-lg"
                  >
                    Upload
                  </a>
                </li>

              </ul>
            </div>
            <div className="flex items-center justify-end">
              <a href="/signup" className="signup-btn font-serif py-3 px-6 text-white rounded-full  text-lg" >Sign Up</a>
              <a href="/login" className="login-btn font-serif py-3 px-6 text-white rounded-full  text-lg">Log In</a>
            </div>
          </div>
        </nav>
      </div>
    </header>
  );
};

export default Header;







