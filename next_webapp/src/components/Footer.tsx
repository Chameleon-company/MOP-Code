import React from 'react';

const Footer = () => {
    return (
        <footer className="bg-white text-black justify-center">
            <div className="w-full px-32 py-7 flex items-center">
                <div className="flex-shrink-0 mr-4">
                    <img src='/img/header-logo.png' alt='Logo' className='w-20' />
                </div>
                <div className="flex-grow text-center">
                    <ul className="flex justify-center gap-20">
                        <li><a href='/privacypolicy'>Privacy Policy</a></li>
                        <li><a href='/licensing'>Licensing</a></li>
                        <li><a href='/contact'>Contact Us</a></li>
                    </ul>
                </div>
                <div className="flex-shrink-0 ml-4 flex gap-2">
                    <a href='https://data.melbourne.vic.gov.au/pages/home/'>Melbourne Open Data</a>
                    <img src='https://www.svgrepo.com/show/510970/external-link.svg' alt='Link Icon' className='w-5' />
                </div>
            </div>
        </footer>
    );
};

export default Footer;
