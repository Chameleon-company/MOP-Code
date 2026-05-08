"use client";

import React, { useState, useEffect, useRef } from "react";
import { useTranslations } from "next-intl";
import {
	Link,
	useRouter as useI18nRouter,
	usePathname as useI18nPathname,
} from "@/i18n-navigation";
import LanguageDropdown from "./LanguageDropdown";
import { HiMenu, HiX, HiMoon, HiSun, HiChevronDown } from "react-icons/hi";
import { useTheme } from "../hooks/useTheme";
import { usePathname } from "next/navigation";

const languages = [
	{ name: "English", locale: "en" },
	{ name: "Chinese (中文)", locale: "cn" },
	{ name: "Spanish (Español)", locale: "es" },
	{ name: "Greek (Ελληνικά)", locale: "el" },
	{ name: "Arabic (العربية)", locale: "ar" },
	{ name: "Italian (Italiano)", locale: "it" },
	{ name: "Hindi (हिन्दी)", locale: "hi" },
	{ name: "Vietnamese (Tiếng Việt)", locale: "vi" },
] as const;

const navItems = [
	{ type: "link", name: "Home", link: "/" },
	{ type: "link", name: "About Us", link: "/about" },
	{ type: "link", name: "Profile", link: "/profile" },
	{
		type: "dropdown",
		name: "Explore",
		items: [
			{ name: "Use Cases", link: "/usecases" },
			{ name: "Blogs", link: "/blog" },
			{ name: "Gallery", link: "/gallery" },
			{ name: "Contact Us", link: "/contact" },
		],
	},
] as const;

const Header = () => {
	const t = useTranslations("common");
	const pathname = usePathname();
	const i18nPathname = useI18nPathname();
	const i18nRouter = useI18nRouter();
	const exploreRef = useRef<HTMLDivElement | null>(null);

	const [isMenuOpen, setIsMenuOpen] = useState(false);
	const [openDropdown, setOpenDropdown] = useState<string | null>(null);
	const [openMobileDropdown, setOpenMobileDropdown] = useState<string | null>(null);
	const [isLangOpen, setIsLangOpen] = useState(false);
	const [isLoggedIn, setIsLoggedIn] = useState(false);
	const { theme, toggleTheme } = useTheme();

	useEffect(() => {
		const token = localStorage.getItem("token");
		setIsLoggedIn(!!token);
	}, []);

	const handleLogout = () => {
		localStorage.removeItem("token");
		localStorage.removeItem("user");
		localStorage.removeItem("userId");
		setIsLoggedIn(false);
		i18nRouter.push("/");
	};

	const selectLanguage = (locale: string) => {
		i18nRouter.push(i18nPathname, { locale });
		i18nRouter.refresh();
		setIsLangOpen(false);
		closeMenu();
	};

	const toggleMenu = () => setIsMenuOpen((prev) => !prev);

	const closeMenu = () => {
		setIsMenuOpen(false);
		setOpenMobileDropdown(null);
		setIsLangOpen(false);
	};

	const toggleMobileDropdown = (name: string) =>
		setOpenMobileDropdown((prev) => (prev === name ? null : name));

	const toggleDesktopDropdown = (name: string) =>
		setOpenDropdown((prev) => (prev === name ? null : name));

	const isActiveLink = (link: string) => {
		const cleanPath = pathname.replace(/^\/[a-z]{2}(?=\/|$)/, "") || "/";
		if (link === "/") return cleanPath === "/";
		return cleanPath === link || cleanPath.startsWith(`${link}/`);
	};

	const isDropdownActive = (items: readonly { name: string; link: string }[]) =>
		items.some((sub) => isActiveLink(sub.link));

	const navLinkBase =
		"px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-green-500";

	const navLinkActive =
		"bg-gradient-to-r from-green-500 to-emerald-600 text-white shadow-md shadow-green-500/25";

	const navLinkIdle =
		"text-gray-700 hover:text-green-700 hover:bg-green-50 dark:text-gray-200 dark:hover:text-green-300 dark:hover:bg-green-900/20";

	const dropdownPanel =
		"absolute left-0 top-full mt-2 z-[100] w-52 overflow-hidden rounded-2xl border border-gray-100 bg-white shadow-2xl dark:border-gray-700 dark:bg-gray-900";

	const dropdownItem = (active: boolean) =>
		`flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition-all duration-150 ${
			active
				? "bg-gradient-to-r from-green-500 to-emerald-600 text-white"
				: "text-gray-700 hover:bg-green-50 hover:text-green-700 dark:text-gray-200 dark:hover:bg-green-900/25 dark:hover:text-green-300"
		}`;

	return (
		<header className="sticky top-0 z-50 bg-white/85 backdrop-blur-md border-b border-gray-200/60 shadow-sm dark:bg-gray-900/85 dark:border-gray-700/50">
			<link
				href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap"
				rel="stylesheet"
			/>

			<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
				<div className="flex items-center justify-between h-16 w-full">
					{/* Logo + Desktop Nav */}
					<div className="flex items-center min-w-0">
						<Link
							href="/"
							className="flex-shrink-0 flex items-center"
							aria-label="Go to homepage"
						>
							<img
								className="h-16 w-auto transition-transform duration-300 ease-out hover:scale-110 hover:drop-shadow-lg"
								src="/img/new-logo-green.png"
								alt="Logo"
							/>
						</Link>

						<nav
							className="ml-8 hidden lg:flex lg:items-center gap-2"
							aria-label="Main navigation"
						>
							{navItems.map((item) =>
								item.type === "link" ? (
									<Link
										key={item.name}
										href={item.link}
										className={`${navLinkBase} ${
											isActiveLink(item.link) ? navLinkActive : navLinkIdle
										}`}
									>
										{item.name}
									</Link>
								) : (
									<div key={item.name} ref={exploreRef} className="relative">
										<button
											type="button"
											aria-haspopup="true"
											aria-expanded={openDropdown === item.name}
											onClick={() => toggleDesktopDropdown(item.name)}
											className={`${navLinkBase} flex items-center gap-1 ${
												isDropdownActive(item.items) ? navLinkActive : navLinkIdle
											}`}
										>
											{item.name}
											<HiChevronDown
												className={`h-4 w-4 transition-transform duration-300 ${
													openDropdown === item.name ? "rotate-180" : ""
												}`}
											/>
										</button>

										{openDropdown === item.name && (
											<div className={dropdownPanel} role="menu">
												<div className="h-0.5 w-full bg-gradient-to-r from-green-500 to-emerald-500" />

												<div className="py-1.5">
													{item.items.map((sub) => (
														<Link
															key={sub.name}
															href={sub.link}
															role="menuitem"
															onClick={() => setOpenDropdown(null)}
															className={dropdownItem(isActiveLink(sub.link))}
														>
															{sub.name}
														</Link>
													))}
												</div>
											</div>
										)}
									</div>
								)
							)}
						</nav>
					</div>

					{/* Right-side actions */}
					<div className="flex items-center gap-1.5">
						<button
							onClick={toggleTheme}
							aria-label={
								theme === "dark" ? "Switch to light mode" : "Switch to dark mode"
							}
							className="p-2 rounded-full text-gray-500 hover:text-gray-900 hover:bg-gray-100 dark:text-gray-400 dark:hover:text-white dark:hover:bg-gray-800 transition-all duration-200 hover:scale-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-green-500 focus-visible:ring-offset-2 dark:focus-visible:ring-offset-gray-900"
						>
							{theme === "dark" ? (
								<HiSun className="h-5 w-5" />
							) : (
								<HiMoon className="h-5 w-5" />
							)}
						</button>

						<div className="hidden lg:block">
							<LanguageDropdown />
						</div>

						<div className="hidden lg:flex">
							{isLoggedIn ? (
								<button
									onClick={handleLogout}
									className="inline-flex h-10 min-w-[96px] items-center justify-center rounded-lg border border-green-600 bg-white px-4 text-sm font-medium text-green-600 transition-all duration-200 transform hover:scale-105 hover:bg-green-50 hover:text-green-700 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 dark:border-green-400 dark:bg-black dark:text-green-300 dark:hover:bg-gray-800"
								>
									Log Out
								</button>
							) : (
								<Link
									href="/login"
									className="inline-flex h-10 min-w-[96px] items-center justify-center rounded-lg border border-green-600 bg-white px-4 text-sm font-medium text-green-600 transition-all duration-200 transform hover:scale-105 hover:bg-green-50 hover:text-green-700 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 dark:border-green-400 dark:bg-black dark:text-green-300 dark:hover:bg-gray-800"
								>
									{t("Log In")}
								</Link>
							)}
						</div>

						<div className="flex lg:hidden">
							<button
								onClick={toggleMenu}
								aria-label={isMenuOpen ? "Close menu" : "Open menu"}
								aria-expanded={isMenuOpen}
								className="p-2 rounded-full text-gray-500 hover:text-gray-900 hover:bg-gray-100 dark:text-gray-400 dark:hover:text-white dark:hover:bg-gray-800 transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-green-500 focus-visible:ring-offset-2 dark:focus-visible:ring-offset-gray-900"
							>
								{isMenuOpen ? (
									<HiX className="h-6 w-6" />
								) : (
									<HiMenu className="h-6 w-6" />
								)}
							</button>
						</div>
					</div>
				</div>

				{/* Mobile Menu */}
				{isMenuOpen && (
					<div className="lg:hidden pb-4">
						<nav
							className="mt-2 space-y-1 rounded-2xl border border-gray-100 bg-white p-3 shadow-2xl dark:border-gray-700 dark:bg-gray-900"
							aria-label="Mobile navigation"
						>
							<div className="h-0.5 w-full rounded-full bg-gradient-to-r from-green-500 to-emerald-500 mb-2" />

							{navItems.map((item) =>
								item.type === "link" ? (
									<Link
										key={item.name}
										href={item.link}
										onClick={closeMenu}
										className={`block px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
											isActiveLink(item.link)
												? "bg-gradient-to-r from-green-500 to-emerald-600 text-white shadow-sm"
												: "text-gray-700 hover:text-green-700 hover:bg-green-50 dark:text-gray-200 dark:hover:text-green-300 dark:hover:bg-green-900/20"
										}`}
									>
										{item.name === "Home" || item.name === "About Us"
											? t(item.name)
											: item.name}
									</Link>
								) : (
									<div key={item.name}>
										<button
											type="button"
											aria-expanded={openMobileDropdown === item.name}
											onClick={() => toggleMobileDropdown(item.name)}
											className={`w-full flex items-center justify-between px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
												isDropdownActive(item.items)
													? "bg-gradient-to-r from-green-500 to-emerald-600 text-white shadow-sm"
													: "text-gray-700 hover:text-green-700 hover:bg-green-50 dark:text-gray-200 dark:hover:text-green-300 dark:hover:bg-green-900/20"
											}`}
										>
											<span>{item.name}</span>
											<HiChevronDown
												className={`h-4 w-4 transition-transform duration-300 ${
													openMobileDropdown === item.name ? "rotate-180" : ""
												}`}
											/>
										</button>

										{openMobileDropdown === item.name && (
											<div className="mt-1 ml-3 space-y-0.5 border-l-2 border-green-200 pl-3 dark:border-green-800/60">
												{item.items.map((sub) => (
													<Link
														key={sub.name}
														href={sub.link}
														onClick={closeMenu}
														className={`block px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
															isActiveLink(sub.link)
																? "bg-gradient-to-r from-green-500 to-emerald-600 text-white shadow-sm"
																: "text-gray-700 hover:text-green-700 hover:bg-green-50 dark:text-gray-200 dark:hover:text-green-300 dark:hover:bg-green-900/20"
														}`}
													>
														{sub.name}
													</Link>
												))}
											</div>
										)}
									</div>
								)
							)}

							<div>
								<button
									type="button"
									aria-expanded={isLangOpen}
									onClick={() => setIsLangOpen((prev) => !prev)}
									className="w-full flex items-center justify-between px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 text-gray-700 hover:text-green-700 hover:bg-green-50 dark:text-gray-200 dark:hover:text-green-300 dark:hover:bg-green-900/20"
								>
									<span>{t("Language")}</span>
									<HiChevronDown
										className={`h-4 w-4 transition-transform duration-300 ${
											isLangOpen ? "rotate-180" : ""
										}`}
									/>
								</button>

								{isLangOpen && (
									<div className="mt-1 ml-3 space-y-0.5 border-l-2 border-green-200 pl-3 dark:border-green-800/60">
										{languages.map((lang) => (
											<button
												key={lang.locale}
												onClick={() => selectLanguage(lang.locale)}
												className="block w-full text-left px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 text-gray-700 hover:text-green-700 hover:bg-green-50 dark:text-gray-200 dark:hover:text-green-300 dark:hover:bg-green-900/20"
											>
												{lang.name}
											</button>
										))}
									</div>
								)}
							</div>

							{isLoggedIn ? (
								<button
									onClick={handleLogout}
									className="block w-full text-left text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-base font-medium"
								>
									Log Out
								</button>
							) : (
								<Link
									href="/login"
									className="block text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-base font-medium"
								>
									{t("Log In")}
								</Link>
							)}
						</nav>
					</div>
				)}
			</div>
		</header>
	);
};

export default Header;