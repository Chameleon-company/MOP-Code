"use client";

import React, { useState, useEffect, useCallback } from "react";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import SearchBar, { LocalSearchMode } from "./searchbar";
import PreviewComponent from "./preview";
import { CATEGORY, CaseStudy } from "../../types";
import Tooglebutton from "../Tooglebutton/Tooglebutton";

const PAGE_SIZE = 9;

const UseCases: React.FC = () => {
	const [darkMode, setDarkMode] = useState(false);
	const [usecases, setUsecases] = useState<CaseStudy[]>([]);
	const [loading, setLoading] = useState(true);
	const [page, setPage] = useState(1);
	const [totalPages, setTotalPages] = useState(1);
	const [total, setTotal] = useState(0);
	const [searchTerm, setSearchTerm] = useState("");
	const [searchMode, setSearchMode] = useState<LocalSearchMode>("title");

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

	const fetchUsecases = useCallback(async (currentPage: number, term: string, mode: LocalSearchMode) => {
		setLoading(true);
		try {
			const params = new URLSearchParams({
				page: String(currentPage),
				pageSize: String(PAGE_SIZE),
			});

			if (term) {
				if (mode === "tag") {
					params.set("tag_name", term);
				} else {
					params.set("search", term);
					params.set("search_by", mode === "content" ? "description" : "title");
				}
			}

			const res = await fetch(`/api/usecases?${params}`);
			const json = await res.json();

			if (json.success) {
				const mapped: CaseStudy[] = (json.data || []).map((u: any) => ({
					id: u.id,
					title: u.title,
					description: u.description ?? "",
					tags: (u.tags || []).map((t: any) => (typeof t === "string" ? t : t.name)),
					image: u.cover_img ?? "",
					category: u.category_id ? String(u.category_id) : "",
					htmlFile: "",
				}));
				setUsecases(mapped);
				setTotal(json.pagination?.total ?? 0);
				setTotalPages(json.pagination?.totalPages ?? 1);
			}
		} catch (e) {
			console.error(e);
		} finally {
			setLoading(false);
		}
	}, []);

	useEffect(() => {
		fetchUsecases(page, searchTerm, searchMode);
	}, [page, searchTerm, searchMode, fetchUsecases]);

	const handleSearch = useCallback((term: string, mode: LocalSearchMode, _cat: CATEGORY) => {
		setSearchTerm(term);
		setSearchMode(mode);
		setPage(1);
	}, []);

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

					<SearchBar onSearch={handleSearch} />

					{loading ? (
						<div className="flex justify-center py-20">
							<div className="h-8 w-8 animate-spin rounded-full border-4 border-green-500 border-t-transparent" />
						</div>
					) : (
						<PreviewComponent
							caseStudies={usecases}
							page={page}
							totalPages={totalPages}
							total={total}
							onPageChange={setPage}
						/>
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
