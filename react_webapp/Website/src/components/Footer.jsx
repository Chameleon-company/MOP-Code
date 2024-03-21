// Footer.js
import React from 'react';
import { Link } from 'react-router-dom';

const Footer = () => {
  return (
    <footer class="bg-white rounded-lg shadow dark:bg-gray-900 m-4">
    <div class="w-full max-w-screen-xl mx-auto p-4 md:py-8">
        <div class="sm:flex sm:items-center sm:justify-between">
            <a href="" class="flex items-center mb-4 sm:mb-0 space-x-3 rtl:space-x-reverse">
                <img src="src/assets/react.svg" class="h-8" alt="MOP logo" />
                <span class="self-center text-2xl font-semibold whitespace-nowrap dark:text-white">Melbourne Open Playground</span>
            </a>
            <ul class="flex flex-wrap items-center mb-6 text-sm font-medium text-gray-500 sm:mb-0 dark:text-gray-400">
                <li>
                    <Link to="/about" class="hover:underline me-4 md:me-6">About</Link>
                </li>
                <li>
                    <Link to="/privacypolicy" class="hover:underline me-4 md:me-6">Privacy Policy</Link>
                </li>
                <li>
                    <Link to="/licensing" class="hover:underline me-4 md:me-6">Licensing</Link>
                </li>
                <li>
                    <Link to="/contact" class="hover:underline">Contact</Link>
                </li>
            </ul>
        </div>
        <hr class="my-6 border-gray-200 sm:mx-auto dark:border-gray-700 lg:my-8" />
        <span class="block text-sm text-gray-500 sm:text-center dark:text-gray-400">© 2023 <a href="" class="hover:underline"></a>. All Rights Reserved.</span>
    </div>
</footer>
  );
};

export default Footer;
