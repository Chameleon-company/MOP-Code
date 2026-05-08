"use client";
import { usePathname, useRouter } from "@/i18n-navigation";
import React, { useState, useRef, useEffect } from "react";
import { useTranslations } from "next-intl";
import { HiChevronDown } from "react-icons/hi";

const languages = [
	{ name: "English", locale: "en" },
	{ name: "Chinese (中文)", locale: "cn" },
	{ name: "Spanish (Español)", locale: "es" },
	{ name: "Greek (Ελληνικά)", locale: "el" },
	{ name: "Arabic (العربية)", locale: "ar" },
	{ name: "Italian (Italiano)", locale: "it" },
	{ name: "Hindi (हिन्दी)", locale: "hi" },
	{ name: "Vietnamese (Tiếng Việt)", locale: "vi" },
];

const LanguageDropdown: React.FC = () => {
	const [isOpen, setIsOpen] = useState<boolean>(false);
	const ref = useRef<HTMLDivElement>(null);
	const router = useRouter();
	const pathname = usePathname();
	const t = useTranslations("common");

	// Close on outside click
	useEffect(() => {
		const onClickOutside = (e: MouseEvent) => {
			if (ref.current && !ref.current.contains(e.target as Node)) {
				setIsOpen(false);
			}
		};
		document.addEventListener("mousedown", onClickOutside);
		return () => document.removeEventListener("mousedown", onClickOutside);
	}, []);

	// Close on Escape
	useEffect(() => {
		const onEscape = (e: KeyboardEvent) => {
			if (e.key === "Escape") setIsOpen(false);
		};
		document.addEventListener("keydown", onEscape);
		return () => document.removeEventListener("keydown", onEscape);
	}, []);

	const selectLanguage = (locale: string) => {
		setIsOpen(false);
		router.push(pathname, { locale });
		router.refresh();
	};

	return (
		<div ref={ref} className="relative">
			{/* Trigger button */}
			<button
				type="button"
				onClick={() => setIsOpen((prev) => !prev)}
				aria-expanded={isOpen}
				aria-haspopup="listbox"
				aria-label="Select language"
				className={`inline-flex h-9 items-center justify-center gap-1.5 rounded-full border px-4 text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-green-500 focus-visible:ring-offset-2 dark:focus-visible:ring-offset-gray-900 ${
					isOpen
						? "border-green-500 bg-green-50 text-green-700 dark:border-green-500 dark:bg-green-900/25 dark:text-green-300"
						: "border-green-500/50 text-green-600 hover:border-green-500 hover:bg-green-50 hover:text-green-700 hover:-translate-y-0.5 hover:shadow-sm dark:border-green-500/40 dark:text-green-400 dark:hover:bg-green-900/20 dark:hover:text-green-300"
				}`}
			>
				{t("Language")}
				<HiChevronDown
					className={`h-3.5 w-3.5 transition-transform duration-300 ${
						isOpen ? "rotate-180" : ""
					}`}
				/>
			</button>

			{/* Dropdown panel – solid card for readability on any background */}
			{isOpen && (
				<div
					role="listbox"
					aria-label="Language options"
					className="absolute right-0 top-full mt-2 z-[100] w-56 overflow-hidden rounded-2xl border border-gray-100 bg-white shadow-2xl dark:border-gray-700 dark:bg-gray-900"
				>
					{/* Green accent top bar */}
					<div className="h-0.5 w-full bg-gradient-to-r from-green-500 to-emerald-500" />
					<div className="py-1.5">
						{languages.map((lang) => (
							<button
								key={lang.locale}
								role="option"
								onClick={() => selectLanguage(lang.locale)}
								className="flex w-full items-center gap-2 px-4 py-2.5 text-sm font-medium text-gray-700 transition-all duration-150 hover:bg-green-50 hover:text-green-700 dark:text-gray-200 dark:hover:bg-green-900/25 dark:hover:text-green-300"
							>
								{lang.name}
							</button>
						))}
					</div>
				</div>
			)}
		</div>
	);
};

export default LanguageDropdown;
