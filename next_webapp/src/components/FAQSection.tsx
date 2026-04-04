"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChevronDown,
  HelpCircle,
  Search,
  Database,
  Upload,
  Bot,
  Moon,
} from "lucide-react";

type FAQItem = {
  question: string;
  answer: string;
  icon: React.ReactNode;
};

const faqs: FAQItem[] = [
  {
    icon: <HelpCircle size={15} />,
    question: "What is MOP (Melbourne Open Platform)?",
    answer:
      "MOP is an open data platform that aggregates and visualises urban data for Melbourne. It provides access to use cases, datasets, and insights that help researchers, policymakers, and the public understand city trends.",
  },
  {
    icon: <Search size={15} />,
    question: "How do I search for use cases?",
    answer:
      "Use the search bar on the home page to find use cases by keyword or category. You can also browse through the Use Cases section to filter by topic, tag, or data type.",
  },
  {
    icon: <Database size={15} />,
    question: "Is the data on MOP free to access?",
    answer:
      "Yes, all data and use cases published on MOP are freely accessible to the public. Some datasets may link to external sources with their own licensing terms — check the individual use case page for details.",
  },
  {
    icon: <Upload size={15} />,
    question: "Can I contribute a use case or dataset?",
    answer:
      "Absolutely. Registered users can upload new use cases via the Upload page. Make sure your submission includes a clear description, relevant tags, and a link to the source data.",
  },
  {
    icon: <Bot size={15} />,
    question: "How do I use the AI chatbot?",
    answer:
      "Click the chat icon in the bottom-right corner to open the chatbot. You can ask it to navigate to pages, search use cases by keyword, or answer general questions about the platform. Voice input is also supported.",
  },
  {
    icon: <Moon size={15} />,
    question: "Does MOP support dark mode?",
    answer:
      "Yes! MOP supports both light and dark mode. Use the theme toggle in the navigation bar to switch between them. Your preference is remembered across sessions.",
  },
];

export default function FAQSection() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const toggle = (index: number) =>
    setOpenIndex((prev) => (prev === index ? null : index));

  return (
    <section className="w-full bg-white dark:bg-[#1f2a30] py-12 sm:py-20 px-4 sm:px-6">
      <div className="max-w-3xl mx-auto">

        {/* ── Section header ────────────────────────────────────────────── */}
        <div className="text-center mb-6">
          <span className="inline-block mb-4 px-3 py-1 rounded-full text-xs font-semibold tracking-widest uppercase bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400">
            FAQ
          </span>
          <h2 className="text-2xl sm:text-4xl font-bold tracking-tight text-gray-900 dark:text-white mb-3">
            Frequently Asked Questions
          </h2>
          <p className="text-gray-500 dark:text-gray-400 text-sm sm:text-base max-w-xl mx-auto leading-relaxed">
            Everything you need to know about the Melbourne Open Platform.
          </p>
        </div>

        {/* ── Accordion list ────────────────────────────────────────────── */}
        <div className="flex flex-col gap-3">
          {faqs.map((faq, index) => {
            const isOpen = openIndex === index;
            const panelId = `faq-panel-${index}`;
            const triggerId = `faq-trigger-${index}`;

            return (
              <div
                key={index}
                className={[
                  "rounded-xl sm:rounded-2xl border overflow-hidden bg-white dark:bg-[#263238]",
                  "transition-all duration-300",
                  isOpen
                    ? "border-emerald-400 dark:border-emerald-500 shadow-[0_8px_32px_rgba(52,211,153,0.25)] dark:shadow-[0_8px_32px_rgba(52,211,153,0.30)]"
                    : "border-gray-200 dark:border-white/10 shadow-md hover:shadow-xl hover:border-gray-300 dark:hover:bg-[#2d3d45] dark:hover:border-emerald-500/50 dark:hover:shadow-[0_8px_28px_rgba(52,211,153,0.22)]",
                ].join(" ")}
              >
                {/* ── Question button ──────────────────────────────────── */}
                <button
                  id={triggerId}
                  aria-controls={panelId}
                  aria-expanded={isOpen}
                  onClick={() => toggle(index)}
                  className="w-full flex items-center gap-3 sm:gap-4 px-4 sm:px-6 py-4 sm:py-5 text-left group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-inset"
                >
                  {/* Icon badge */}
                  <span
                    aria-hidden="true"
                    className={[
                      "shrink-0 flex items-center justify-center w-7 h-7 sm:w-8 sm:h-8 rounded-lg transition-all duration-200",
                      isOpen
                        ? "bg-emerald-500 text-white shadow-sm"
                        : "bg-emerald-100 text-emerald-600 dark:bg-emerald-500/15 dark:text-emerald-400 group-hover:bg-emerald-200 dark:group-hover:bg-emerald-500/25",
                    ].join(" ")}
                  >
                    {faq.icon}
                  </span>

                  {/* Question text */}
                  <span
                    className={[
                      "flex-1 text-[13px] sm:text-[15px] font-semibold leading-snug transition-colors duration-200",
                      isOpen
                        ? "text-emerald-600 dark:text-emerald-400"
                        : "text-gray-800 dark:text-gray-100 group-hover:text-emerald-600 dark:group-hover:text-emerald-400",
                    ].join(" ")}
                  >
                    {faq.question}
                  </span>

                  {/* Chevron — spring rotation */}
                  <motion.span
                    aria-hidden="true"
                    animate={{ rotate: isOpen ? 180 : 0 }}
                    transition={{ type: "spring", stiffness: 300, damping: 25 }}
                    className={[
                      "shrink-0 transition-colors duration-200",
                      isOpen
                        ? "text-emerald-500"
                        : "text-gray-400 dark:text-gray-500 group-hover:text-emerald-500",
                    ].join(" ")}
                  >
                    <ChevronDown size={16} className="sm:hidden" />
                    <ChevronDown size={18} className="hidden sm:block" />
                  </motion.span>
                </button>

                {/* ── Thin divider (only while open) ──────────────────── */}
                <AnimatePresence initial={false}>
                  {isOpen && (
                    <motion.div
                      key="divider"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.15 }}
                      className="mx-4 sm:mx-6 h-px bg-emerald-100 dark:bg-emerald-500/20"
                    />
                  )}
                </AnimatePresence>

                {/* ── Answer panel ─────────────────────────────────────── */}
                <AnimatePresence initial={false}>
                  {isOpen && (
                    <motion.div
                      id={panelId}
                      role="region"
                      aria-labelledby={triggerId}
                      key="answer"
                      initial={{ height: 0, opacity: 0, y: -6 }}
                      animate={{ height: "auto", opacity: 1, y: 0 }}
                      exit={{ height: 0, opacity: 0, y: -6 }}
                      transition={{ duration: 0.28, ease: [0.4, 0, 0.2, 1] }}
                      className="overflow-hidden"
                    >
                      <p className="px-4 sm:px-6 pt-3 sm:pt-4 pb-4 sm:pb-6 text-xs sm:text-sm text-gray-600 dark:text-gray-300 leading-7">
                        {faq.answer}
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            );
          })}
        </div>

      </div>
    </section>
  );
}
