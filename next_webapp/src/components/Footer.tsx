'use client';
import React, { useState } from 'react';
import { Link } from "@/i18n-navigation";
import { useTranslations } from "next-intl";
import { FaFacebook, FaLinkedin } from 'react-icons/fa';
import { FaSquareXTwitter } from "react-icons/fa6";

const Footer = () => {
  const t = useTranslations("common");

  // Feedback form state
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [feedbackStatus, setFeedbackStatus] = useState<'idle' | 'error' | 'success'>('idle');
  const [feedbackMsg, setFeedbackMsg] = useState('');

  const handleFeedbackSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/) || !message.trim()) {
      setFeedbackStatus('error');
      setFeedbackMsg('Please fill out all fields with valid information.');
      return;
    }
    // Simulate success
    setFeedbackStatus('success');
    setFeedbackMsg('Thanks for your feedback!');
    setName('');
    setEmail('');
    setMessage('');
  };

  return (
    <footer className="bg-green-500 dark:bg-green-800 text-white pt-10 pb-6">
      <div className="container mx-auto px-4 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-10">

          {/* Logo & Navigation */}
          <div className="flex flex-col items-center md:items-start space-y-4">
            <img src="/img/new-logo-white.png" alt="Hameleon logo" className="h-16 lg:h-24 w-auto" />
            <nav>
              <ul className="flex flex-col items-center md:items-start space-y-2 text-base lg:text-lg">
                <li>
                  <Link href="/licensing" className="hover:underline">
                    {t("Licensing")}
                  </Link>
                </li>
                <li>
                  <Link href="/privacypolicy" className="hover:underline">
                    {t("Privacy Policy")}
                  </Link>
                </li>
                <li>
                  <Link href="/contact" className="hover:underline">
                    {t("Contact Us")}
                  </Link>
                </li>
              </ul>
            </nav>
          </div>

          {/* Newsletter + Feedback Form */}
          <div className="space-y-4 text-center md:text-left">
            <h3 className="text-lg font-semibold">Subscribe to Our Newsletter</h3>
            <p className="text-sm">Stay updated with the latest from Chameleon.</p>
            <form onSubmit={(e) => e.preventDefault()} className="flex flex-col sm:flex-row gap-2">
              <input
                type="email"
                placeholder="Your email"
                className="px-4 py-2 text-black rounded w-full sm:w-auto"
                required
              />
              <button
                type="submit"
                className="bg-white text-green-600 hover:bg-green-100 px-4 py-2 rounded font-semibold"
              >
                Subscribe
              </button>
            </form>

            {/* Feedback Form */}
            <h3 className="text-lg font-semibold mt-6">Feedback</h3>
            <form onSubmit={handleFeedbackSubmit} className="flex flex-col gap-2">
              <input
                type="text"
                placeholder="Your Name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="px-4 py-2 text-black rounded"
                required
              />
              <input
                type="email"
                placeholder="Your Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="px-4 py-2 text-black rounded"
                required
              />
              <textarea
                placeholder="Your Message"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                rows={3}
                className="px-4 py-2 text-black rounded"
                required
              />
              <button
                type="submit"
                className="bg-white text-green-600 hover:bg-green-100 px-4 py-2 rounded font-semibold"
              >
                Send
              </button>
              {feedbackStatus !== 'idle' && (
                <p className={`text-sm ${feedbackStatus === 'error' ? 'text-red-300' : 'text-green-300'}`}>
                  {feedbackMsg}
                </p>
              )}
            </form>
          </div>

          {/* Social Media + External Link */}
          <div className="space-y-4 text-center md:text-left">
            <h3 className="text-lg font-semibold">Connect with Us</h3>
            <div className="flex justify-center md:justify-start gap-4">
              <a href="#" aria-label="Facebook" className="hover:text-gray-300 transition">
                <FaFacebook size={24} />
              </a>
              <a href="#" aria-label="Twitter" className="hover:text-gray-300 transition">
                <FaSquareXTwitter size={24} />
              </a>
              <a href="#" aria-label="LinkedIn" className="hover:text-gray-300 transition">
                <FaLinkedin size={24} />
              </a>
            </div>
            <a
              href="https://data.melbourne.vic.gov.au/pages/home/"
              className="inline-flex items-center justify-center hover:underline text-sm mt-2"
              target="_blank" rel="noreferrer"
            >
              Melbourne Open Data
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" viewBox="0 0 20 20" fill="currentColor">
                <path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z" />
                <path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z" />
              </svg>
            </a>
          </div>
        </div>

        {/* Footer Bottom */}
        <div className="text-center text-sm mt-10 border-t border-white/30 pt-4">
          © {new Date().getFullYear()} Chameleon Company. All rights reserved.
        </div>
      </div>
    </footer>
  );
};

export default Footer;
