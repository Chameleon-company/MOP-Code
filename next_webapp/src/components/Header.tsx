"use client";
import React, { useState } from "react";
import { useTranslations } from "next-intl";
import { Link } from "@/i18n-navigation";
import LanguageDropdown from "./LanguageDropdown";
import { HiMenu, HiX, HiMoon, HiSun, HiChevronDown } from "react-icons/hi";
import { useTheme } from "../hooks/useTheme";
import { usePathname } from "next/navigation";

const Header = () => {
	const t = useTranslations("common");
	const pathname = usePathname();
	const [isMenuOpen, setIsMenuOpen] = useState(false);
	const [openDropdown, setOpenDropdown] = useState<string | null>(null);
    const [openMobileDropdown, setOpenMobileDropdown] = useState<string | null>(null);
	const { theme, toggleTheme } = useTheme();

	const toggleMenu = () => {
		setIsMenuOpen(!isMenuOpen);
	};
	const closeMenu = () => {
		setIsMenuOpen(false);
		setOpenMobileDropdown(null);
	};
	
	const toggleMobileDropdown = (name: string) => {
		setOpenMobileDropdown((prev) => (prev === name ? null : name));
	};

	// Object array for navigation items
	const navItems = [
		{ type: "link", name: "Home", link: "/" },
		{ type: "link", name: "About Us", link: "/about" },
		{ type: "link", name: "Upload", link: "/upload" },
		{ type: "link", name: "Profile", link: "/profile" }, 
		{ type: "link", name: "Stastics", link: "/stastics" }, 
		{
			type: "dropdown",
			name: "Explore",
			items: [
				{ name: "Use Cases", link: "/usecases" },
				{ name: "Blogs", link: "/blog" },
				
			],
		},
	] as const;
	const isActiveLink = (link: string) => {
		const cleanPath = pathname.replace(/^\/[a-z]{2}(?=\/|$)/, "") || "/";
	  
		if (link === "/") {
		  return cleanPath === "/";
		}
	  
		return cleanPath === link || cleanPath.startsWith(`${link}/`);
	  };
	  const isDropdownActive = (
		items: readonly { name: string; link: string }[]
	) => {
		return items.some((subItem) => isActiveLink(subItem.link));
	};

	return (
		<header className="sticky top-0 z-50 bg-white shadow-sm border-b border-gray-200 dark:bg-black dark:border-gray-800"> {/* sticky nav-bar */}
			<link
				href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap"
				rel="stylesheet"
			></link>
			<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
			 <div className="flex justify-between items-center h-16 w-full">
					<div className="flex items-center">
						<Link href="/" className="flex-shrink-0">
							<img
								className="h-20 w-auto"
								src="/img/new-logo-green.png"
								alt="Logo"
							/>
						</Link>
						
						{/* Menu Items */}
						{/* Menu Items */}
<nav className="ml-10 hidden lg:flex lg:items-center space-x-4">
	{navItems.map((item) =>
		item.type === "link" ? (
			<Link
				key={item.name}
				href={item.link}
				className={`px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
					isActiveLink(item.link)
						? "text-green-700 bg-green-50 dark:text-green-300 dark:bg-green-900/30"
						: "text-gray-700 hover:text-green-700 hover:bg-green-50 dark:text-gray-200 dark:hover:text-green-300 dark:hover:bg-gray-800"
				}`}
			>
				{item.name}
			</Link>
		) : (
			<div
				key={item.name}
				className="relative"
				onMouseEnter={() => setOpenDropdown(item.name)}
				onMouseLeave={() => setOpenDropdown(null)}
			>
				<button
					type="button"
					className={`flex items-center gap-1 px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
						isDropdownActive(item.items)
							? "text-green-700 bg-green-50 dark:text-green-300 dark:bg-green-900/30"
							: "text-gray-700 hover:text-green-700 hover:bg-green-50 dark:text-gray-200 dark:hover:text-green-300 dark:hover:bg-gray-800"
					}`}
				>
					{item.name}
					<HiChevronDown className="h-4 w-4" />
				</button>

				{openDropdown === item.name && (
						<div className="absolute left-0 top-full w-56 rounded-xl border border-gray-200 bg-white p-2 shadow-lg dark:border-gray-800 dark:bg-black z-50">
						{item.items.map((subItem) => (
							<Link
								key={subItem.name}
								href={subItem.link}
								className={`block px-3 py-2 rounded-md text-sm transition-all duration-200 ${
									isActiveLink(subItem.link)
										? "text-green-700 bg-green-50 dark:text-green-300 dark:bg-green-900/30"
										: "text-gray-700 hover:text-green-700 hover:bg-green-50 dark:text-gray-200 dark:hover:text-green-300 dark:hover:bg-gray-800"
								}`}
							>
								{subItem.name}
							</Link>
						))}
					</div>
				)}
			</div>
		)
	)}
</nav>

						
					</div>
					<div className="flex items-center gap-2">
						<button
							onClick={toggleTheme}
							aria-label="Toggle Dark Mode"
							className="p-1 rounded focus:outline-none"
						>
							{theme === "dark" ? (
								<HiSun className="mr-4 h-5 w-5 text-white" />
							) : (
								<HiMoon className="mr-4 h-5 w-5 text-black" />
							)}
						</button>

						<LanguageDropdown />
						<div className="hidden lg:flex">
							<Link
								href="/signup"
								className=" bg-white text-green-600 hover:bg-gray-50 border border-green-600 px-4 py-2 rounded-xl text-sm font-medium"
							>
								{t("Sign Up")}
							</Link>
							<Link
								href="/login"
								className="ml-4 bg-white text-green-600 hover:bg-gray-50 border border-green-600 px-4 py-2 rounded-xl text-sm font-medium"
							>
								{t("Log In")}
							</Link>
						</div>
						<div className="flex lg:hidden">
		                  <button
			                 onClick={toggleMenu}
			                 className="p-2 rounded-md text-green-600 hover:text-green-900 hover:bg-green-50 dark:text-green-300 dark:hover:bg-gray-800 focus:outline-none transition-all duration-200"
			                 aria-label="Toggle menu"
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
					<nav className="mt-2 space-y-1 rounded-xl border border-gray-200 bg-white p-3 shadow-md dark:border-gray-800 dark:bg-black">
					{navItems.map((item) =>
	item.type === "link" ? (
		<Link
			key={item.name}
			href={item.link}
			onClick={closeMenu}
			className={`block px-3 py-2 rounded-md text-base font-medium transition-all duration-200 ${
				isActiveLink(item.link)
					? "text-green-700 bg-green-50 dark:text-green-300 dark:bg-green-900/20"
					: "text-green-600 hover:text-green-700 hover:bg-green-50 dark:text-gray-200 dark:hover:text-green-300 dark:hover:bg-gray-800"
			}`}
		>
			{item.name === "Home" || item.name === "About Us" ? t(item.name) : item.name}
		</Link>
	) : (
		<div key={item.name}>
			<button
				type="button"
				onClick={() => toggleMobileDropdown(item.name)}
				className={`w-full flex items-center justify-between px-3 py-2 rounded-md text-base font-medium transition-all duration-200 ${
					isDropdownActive(item.items)
						? "text-green-700 bg-green-50 dark:text-green-300 dark:bg-green-900/20"
						: "text-green-600 hover:text-green-700 hover:bg-green-50 dark:text-gray-200 dark:hover:text-green-300 dark:hover:bg-gray-800"
				}`}
			>
				<span>{item.name}</span>
				<HiChevronDown
					className={`h-4 w-4 transition-transform duration-200 ${
						openMobileDropdown === item.name ? "rotate-180" : ""
					}`}
				/>
			</button>

			{openMobileDropdown === item.name && (
				<div className="mt-1 ml-4 space-y-1 border-l border-gray-200 pl-3 dark:border-gray-700">
					{item.items.map((subItem) => (
						<Link
							key={subItem.name}
							href={subItem.link}
							onClick={closeMenu}
							className={`block px-3 py-2 rounded-md text-sm transition-all duration-200 ${
								isActiveLink(subItem.link)
									? "text-green-700 bg-green-50 dark:text-green-300 dark:bg-green-900/20"
									: "text-green-600 hover:text-green-700 hover:bg-green-50 dark:text-gray-200 dark:hover:text-green-300 dark:hover:bg-gray-800"
							}`}
						>
							{subItem.name}
						</Link>
					))}
				</div>
			)}
		</div>
	)
)}
							{/* Add Sign Up and Log In buttons to mobile menu */}
							<Link
								href="/signup"
								className="block text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-base font-medium"
							>
								{t("Sign Up")}
							</Link>
							<Link
								href="/login"
								className="block text-green-600 hover:text-green-900 px-3 py-2 rounded-md text-base font-medium"
							>
								{t("Log In")}
							</Link>
						</nav>
					</div>
				)}
			</div>
		</header>
	);

};

export default Header;
