// Footer.js
"use client"
import React, { useState, useEffect } from 'react';

const Footer = () => {
    const [theme, setTheme] = useState('light'); // Default to light theme

    useEffect(() => {
        const matchDark = window.matchMedia('(prefers-color-scheme: dark)');
        const handleChange = (e: { matches: any; }) => {
            setTheme(e.matches ? 'dark' : 'light');
        };
        // Listen for changes in the theme preference
        matchDark.addEventListener('change', handleChange);

        // Set initial theme based on user preference
        setTheme(matchDark.matches ? 'dark' : 'light');

        // Cleanup function to remove the event listener
        return () => {
            matchDark.removeEventListener('change', handleChange);
        };
    }, []);

    const themeStyles = {
        background: theme === 'dark' ? '#333' : '#fff',
        color: theme === 'dark' ? '#fff' : '#333',
    };

    return (
        <footer style={themeStyles} className="justify-center">
            <div style={{ width: '100%', padding: '32px', display: 'flex', alignItems: 'center' }}>
                <div style={{ marginRight: '16px' }}>
                    <img src='/img/header-logo.png' alt='logo' style={{ width: '80px' }} />
                </div>
                <div style={{ flexGrow: 1, textAlign: 'center' }}>
                    <ul style={{ display: 'flex', justifyContent: 'center', gap: '80px', listStyle: 'none', padding: 0 }}>
                        <a href='/privacypolicy' style={{ color: themeStyles.color }}><li>Privacy Policy</li></a>
                        <a href='/licensing' style={{ color: themeStyles.color }}><li>Licensing</li></a>
                        <a href='/contact' style={{ color: themeStyles.color }}><li>Contact Us</li></a>
                    </ul>
                </div>
                <div style={{ marginLeft: '16px', display: 'flex', gap: '8px' }}>
                    <a href='https://data.melbourne.vic.gov.au/pages/home/' style={{ color: themeStyles.color }}>
                        Melbourne Open Data
                    </a>
                    <img src='https://www.svgrepo.com/show/510970/external-link.svg' alt='link icon' style={{ width: '20px' }} />
                </div>
            </div>
        </footer>
    );
};

export default Footer;