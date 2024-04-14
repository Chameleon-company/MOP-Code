// Footer.tsx
import React from 'react';

const Footer: React.FC = () => {
    return (
        <footer className="bg-white border-t border-gray-200 py-6 mt-8">
            <div className="container mx-auto px-6 flex justify-between items-center">
                <div className="flex items-center">
                    {/* Update with your actual logo and path */}
                    <img src="/img/header-logo.png" alt="Company logo" className="mr-3" />
                    <span className="text-gray-800 text-sm">Your Company</span>
                </div>
                <nav className="flex">
                    {/* Update href attributes with actual paths */}
                    <a href="/licensing" className="text-gray-800 text-sm hover:text-gray-600 mx-2">Licensing</a>
                    <a href="/privacy-policy" className="text-gray-800 text-sm hover:text-gray-600 mx-2">Privacy Policy</a>
                </nav>
                <div>
                    <a href="https://data.melbourne.vic.gov.au" target="_blank" rel="noopener noreferrer" className="text-gray-800 text-sm hover:text-gray-600">
                        Melbourne Open Data <i className="fas fa-external-link-alt"></i>
                    </a>
                </div>
            </div>
        </footer>
    );
};

export default Footer;
