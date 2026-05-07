"use client";

import React, { useState, useEffect, useCallback, useMemo } from "react";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import SearchBar, { LocalSearchMode } from "./searchbar";
import PreviewComponent from "./preview";
import { CATEGORY } from "../../types";
import Tooglebutton from "../Tooglebutton/Tooglebutton";

const PAGE_SIZE = 9;

const UseCases: React.FC = () => {
	const [selectedCaseStudy, setSelectedCaseStudy] = useState<CaseStudy | null>(
		null,
	);

	const [darkMode, setDarkMode] = useState(false);

	const [searchTerm, setSearchTerm] = useState("");
	const [searchMode, setSearchMode] = useState<LocalSearchMode>("title");
	const [selectedCategory, setSelectedCategory] = useState<CATEGORY>(
		"All" as CATEGORY,
	);

	useEffect(() => {
		const storedTheme = localStorage.getItem("theme");

		if (storedTheme === "dark") {
			setDarkMode(true);
			document.documentElement.classList.add("dark");
		}
	}, []);

	useEffect(() => {
		document.documentElement.classList.toggle("dark", darkMode);
	}, [darkMode]);

	const handleToggle = useCallback((value: boolean) => {
		setDarkMode(value);
		localStorage.setItem("theme", value ? "dark" : "light");
	}, []);

	const filteredCaseStudies = useMemo(() => {
		const keyword = searchTerm.trim().toLowerCase();

		if (!keyword) {
			return demoCaseStudies;
		}

		return demoCaseStudies.filter((study) => {
			if (searchMode === "title") {
				return study.name?.toLowerCase().includes(keyword);
			}

			if (searchMode === "tag") {
				return study.tags?.some((tag) => tag.toLowerCase().includes(keyword));
			}

			if (searchMode === "content") {
				return study.description?.toLowerCase().includes(keyword);
			}

			return true;
		});
	}, [searchTerm, searchMode]);

	const handleSearch = useCallback(
		(term: string, mode: LocalSearchMode, cat: CATEGORY) => {
			setSearchTerm(term);
			setSearchMode(mode);
			setSelectedCategory(cat);
			setSelectedCaseStudy(null);
		},
		[],
	);

	const handleSelectCaseStudy = useCallback((caseStudy: CaseStudy) => {
		setSelectedCaseStudy(caseStudy);
	}, []);

	const handleBack = useCallback(() => {
		setSelectedCaseStudy(null);
	}, []);

	const hasCaseStudies = filteredCaseStudies.length > 0;

	return (
		<div className="flex min-h-screen flex-col bg-[#f7f9fb] text-black transition-colors duration-200 dark:bg-gray-900 dark:text-white">
			<Header />

			<main className="flex-grow">
				<div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-10">
					<section className="mb-8 rounded-[28px] border border-gray-200 bg-white px-6 py-8 shadow-sm dark:border-gray-700 dark:bg-gray-800 sm:px-8">
						<div className="mb-4 inline-flex items-center rounded-full bg-green-50 px-4 py-1.5 text-sm font-semibold text-green-700 dark:bg-green-900/30 dark:text-green-300">
							Open Data Use Cases
						</div>

						<div className="max-w-3xl">
							<h1 className="mb-3 text-4xl font-bold tracking-tight sm:text-5xl">
								Use Cases
							</h1>
						</div>
					</section>

					{!selectedCaseStudy && <SearchBar onSearch={handleSearch} />}

					{hasCaseStudies || selectedCaseStudy ? (
						<PreviewComponent
							caseStudies={filteredCaseStudies}
							trendingCaseStudies={filteredCaseStudies}
							selectedCaseStudy={selectedCaseStudy}
							onSelectCaseStudy={handleSelectCaseStudy}
							onBack={handleBack}
						/>
					) : (
						<div className="mt-8 flex min-h-[250px] items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-white px-6 py-12 text-center shadow-sm dark:border-gray-700 dark:bg-gray-800">
							<p className="text-base font-medium text-gray-500 dark:text-gray-400">
								No data available at the moment
							</p>
						</div>
					)}
				</div>
			</main>

			<div className="fixed bottom-4 right-4 z-50">
				<Tooglebutton onValueChange={handleToggle} />
			</div>

			<Footer />
		</div>
	);
};

export default UseCases;
