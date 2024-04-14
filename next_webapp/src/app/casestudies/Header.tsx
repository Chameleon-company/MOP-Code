import React from "react";
import { Link } from "react-router-dom";

const Header: React.FC = () => {
    return (
        <header className="bg-green-600 px-4 sm:px-8 lg:px-16 xl:px-20 2xl:px-40 py-4 flex justify-between items-center">
            <Link to="/" className="flex items-center text-xl font-semibold text-white">
                {/* Adjust the path to your logo */}
                <img src="/case-studies/src/images/Pic.png" alt="Company logo" className="h-12 mr-3" />
                <span>Your Company</span>
            </Link>
            <nav className="hidden md:flex gap-8">
                <Link to="/" className="text-white text-lg hover:text-gray-200 transition-colors">Home</Link>
                <Link to="/about" className="text-white text-lg hover:text-gray-200 transition-colors">About Us</Link>
                <Link to="/" className="text-white text-lg hover:text-gray-200 transition-colors">Case Studies</Link>
                <Link to="/statistics" className="text-white text-lg hover:text-gray-200 transition-colors">Statistics</Link>
                <Link to="/contact" className="text-white text-lg hover:text-gray-200 transition-colors">Contact Us</Link>
            </nav>
            <div className="flex items-center gap-4">
                <Link to="/signup" className="text-green-600 bg-white rounded-full px-6 py-2 text-sm font-semibold hover:text-green-600 transition-all">Sign Up</Link>
                <Link to="/login" className="text-white border-2 border-white rounded-full px-6 py-2 text-sm font-semibold hover:bg-green-700 transition-all">Log In</Link>
            </div>
        </header>
    );
};

export default Header;
