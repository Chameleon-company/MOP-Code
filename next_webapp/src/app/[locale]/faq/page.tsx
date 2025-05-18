// src/app/faqchatbot/faqchatbot.tsx
"use client";

import React from "react";
import Header from "../../../components/Header";
import Chatbot from "../../chatbot/chatbot";
import Footer from "../../../components/Footer";
import { 
  IoHelpCircleOutline, 
  IoNavigate, 
  IoSearch, 
  IoMic 
} from "react-icons/io5";
import styles from "./faqchatbot.module.css";

const FAQ: React.FC = () => (
  <div
    className={`
      ${styles.faqContainer}
      bg-[#F6F9FC] dark:bg-[#263238]
      text-[#263238] dark:text-[#FFFFFF]
      transition-colors duration-300
    `}
  >
    <Header />

    <main className={styles.mainContent}>
      <header className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-[#263238] dark:text-[#FFFFFF]">
          FAQ – Chatbot Help
        </h1>
      </header>

      {/* How to Use */}
      <details
        className={`
          ${styles.section}
          bg-white dark:bg-[#263238]
          border border-gray-200 dark:border-gray-700
          rounded-md mb-6
          transition-colors duration-300
        `}
        open
      >
        <summary
          className="
            flex items-center
            px-4 py-2
            cursor-pointer
            bg-[#F6F9FC] dark:bg-[#2ECC71]
            text-[#263238] dark:text-[#FFFFFF]
            font-semibold
            transition-colors duration-300
          "
        >
          <IoHelpCircleOutline className="inline-block mr-2" />
          How to Use the Chatbot
        </summary>
        <div className={styles.sectionContent}>
          <ul className={styles.list}>
            <li className={styles.listItem}>
              <IoNavigate className={styles.listItemIcon} />
              <span className="dark:text-white">Click the chat icon in the bottom‐right corner to open the widget.</span>
            </li>
            <li className={styles.listItem}>
              <IoSearch className={styles.listItemIcon} />
              <span className="dark:text-white">
                Type or speak your request:
                <ul className="list-disc pl-5 mt-1 space-y-1">
                  <li>Navigation: “Take me to the use‐case page.”</li>
                  <li>Search: “Use case about public transport.”</li>
                  <li>General: “What does MOP stand for?”</li>
                </ul>
              </span>
            </li>
            <li className={styles.listItem}>
              <IoMic className={styles.listItemIcon} />
              <span className="dark:text-white">Click Send icon to submit. Tap the mic icon for voice input.</span>
            </li>
            <li className={styles.listItem}>
              <IoHelpCircleOutline className={styles.listItemIcon} />
              <span className="dark:text-white">Toggle the speaker icon to have the bot read its response aloud.</span>
            </li>
            <li className={styles.listItem}>
              <IoSearch className={styles.listItemIcon} />
              <span className="dark:text-white">
                <strong>Best Live-Search Format:</strong> “use case on &lt;keyword&gt;”
                (e.g. “use case on clustering”).
              </span>
            </li>
          </ul>
        </div>
      </details>

      {/* What’s New */}
      <details
        className={`
          ${styles.section}
          bg-white dark:bg-[#263238]
          border border-gray-200 dark:border-gray-700
          rounded-md mb-6
          transition-colors duration-300
        `}
      >
        <summary
          className="
            flex items-center
            px-4 py-2
            cursor-pointer
            bg-[#F6F9FC] dark:bg-[#2ECC71]
            text-[#263238] dark:text-[#FFFFFF]
            font-semibold
            transition-colors duration-300
          "
        >
          <IoHelpCircleOutline className="inline-block mr-2" />
          What’s New
        </summary>
        <div className={styles.sectionContent}>
          <ul className={styles.list}>
            <li className={styles.listItem}>
              <IoMic className={styles.listItemIcon} />
              <span className="dark:text-white">
                <strong>Voice Commands:</strong> Real-time speech‐to‐text & text‐to‐speech.
              </span>
            </li>
            <li className={styles.listItem}>
              <IoNavigate className={styles.listItemIcon} />
              <span className="dark:text-white">
                <strong>Instant Navigation:</strong> “navigate home”, “navigate about”, etc.
              </span>
            </li>
            <li className={styles.listItem}>
              <IoSearch className={styles.listItemIcon} />
              <span className="dark:text-white">
                <strong>Live Keyword Search:</strong> “use case on…” pulls matching studies on‐the‐fly.
              </span>
            </li>
          </ul>
        </div>
      </details>


      <details
        className={`
          ${styles.section}
          bg-white dark:bg-[#263238]
          border border-gray-200 dark:border-gray-700
          rounded-md
          transition-colors duration-300
        `}
      >
        <summary
          className="
            flex items-center
            px-4 py-2
            cursor-pointer
            bg-[#F6F9FC] dark:bg-[#2ECC71]
            text-[#263238] dark:text-[#FFFFFF]
            font-semibold
            transition-colors duration-300
          "
        >
          <IoHelpCircleOutline className="inline-block mr-2" />
          Common Issues
        </summary>
        <div className={styles.sectionContent}>
          <ul className={styles.list}>
            <li className={styles.listItem}>
              <IoHelpCircleOutline className={styles.listItemIcon} />
              <span className="dark:text-white">Check your internet connection; the chatbot requires network access.</span>
            </li>
            <li className={styles.listItem}>
              <IoHelpCircleOutline className={styles.listItemIcon} />
              <span className="dark:text-white">Refresh the page if the chatbot stops responding.</span>
            </li>
            <li className={styles.listItem}>
              <IoHelpCircleOutline className={styles.listItemIcon} />
              <span className="dark:text-white">
                Use the exact command formats shown above if you don’t get the desired output.
              </span>
            </li>
          </ul>
        </div>
      </details>
    </main>

    <Chatbot />
    <Footer />
  </div>
);

export default FAQ;
