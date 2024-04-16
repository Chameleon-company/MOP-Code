// Footer.js
import React from 'react';
import Link from "next/link";

const Footer = () => {
    return (
        <footer class="bg-white text-black justify-center">
            <div class="w-full px-32 py-7 flex items-center">
                <div class="flex-shrink-0 mr-4">
                    <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRxPnrCwaqIIGAmQxwxtARGH_BrIs4bkTT6wpyax9ObXQ&s' alt='logo' class='w-20' />
                </div>
                <div class="flex-grow text-center">
                    <ul class="flex justify-center">
                        <a href='/privacypolicy'><li class="mr-4">Privacy Policy</li></a>
                        <a href='/licensing'><li>Licensing</li></a>
                    </ul>
                </div>
                <div class="flex-shrink-0 ml-4 flex gap-2">
                    <a href='https://data.melbourne.vic.gov.au/pages/home/'>Melbourne Open Data</a>
                    <img src='https://www.svgrepo.com/show/510970/external-link.svg' alt='link icon'class='w-5'/>
                </div>
            </div>
        </footer>

    );
};

export default Footer;
