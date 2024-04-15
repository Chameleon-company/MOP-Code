import React from "react";

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
                    className="block font-serif py-4 px-5 text-white rounded hover:bg-blue-400 ml-3 text-lg"
                    aria-current="page"
                  >
                    Home
                  </a>
                </li>
                <li className="inline-block">
                  <a
                    href="/about"
                    className="block font-serif py-4 px-5 text-white rounded hover:bg-blue-400 text-lg"
                  >
                    About Us
                  </a>
                </li>
                <li className="inline-block">
                  <a
                    href="/casestudies"
                    className="block font-serif py-4 px-5 text-white rounded hover:bg-blue-400 text-lg"
                  >
                    Case Studies
                  </a>
                </li>
                <li className="inline-block">
                  <a
                    href="/statistics"
                    className="block font-serif py-4 px-5 text-white rounded hover:bg-blue-400 text-lg"
                  >
                    Statistics
                  </a>
                </li>
                <li className="inline-block">
                  <a
                    href="/upload"
                    className="block font-serif py-4 px-5 text-white rounded hover:bg-blue-400 text-lg"
                  >
                    Upload
                  </a>
                </li>
                <li className="inline-block">
                  <a
                    href="/contact"
                    className="block font-serif py-4 px-5 text-white rounded hover:bg-blue-400 text-lg"
                  >
                    Contact Us
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <a href="/signup" className="font-serif py-3 px-6 text-white rounded-full  text-lg " style={{ borderRadius: "20px", backgroundColor: "#4CAF50",  }}>Sign Up</a>
              <a href="/login" className="font-serif py-3 px-6 text-white rounded-full  text-lg" style={{ borderRadius: "20px",  backgroundColor: "#4CAF50", marginLeft: "1px"  }}>Log In</a>
            </div>
          </div>
        </nav>
      </div>
    </header>
  );
};

export default Header;





