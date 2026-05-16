// "use client";

// import React, { useState, useEffect, useCallback } from "react";
// import Header from "../../../components/Header";
// import Footer from "../../../components/Footer";
// import SearchBar, { LocalSearchMode } from "./searchbar";
// import PreviewComponent from "./preview";
// import { CATEGORY, CaseStudy } from "../../types";
// import Tooglebutton from "../Tooglebutton/Tooglebutton";

// const PAGE_SIZE = 9;

// const UseCases: React.FC = () => {
// 	const [darkMode, setDarkMode] = useState(false);
// 	const [usecases, setUsecases] = useState<CaseStudy[]>([]);
// 	const [loading, setLoading] = useState(true);
// 	const [page, setPage] = useState(1);
// 	const [totalPages, setTotalPages] = useState(1);
// 	const [total, setTotal] = useState(0);
// 	const [searchTerm, setSearchTerm] = useState("");
// 	const [searchMode, setSearchMode] = useState<LocalSearchMode>("title");

// 	useEffect(() => {
// 		const storedTheme = localStorage.getItem("theme");
// 		if (storedTheme === "dark") {
// 			setDarkMode(true);
// 			document.documentElement.classList.add("dark");
// 		}
// 	}, []);

// 	useEffect(() => {
// 		document.documentElement.classList.toggle("dark", darkMode);
// 	}, [darkMode]);

// 	const handleToggle = useCallback((value: boolean) => {
// 		setDarkMode(value);
// 		localStorage.setItem("theme", value ? "dark" : "light");
// 	}, []);

// 	const fetchUsecases = useCallback(async (currentPage: number, term: string, mode: LocalSearchMode) => {
// 		setLoading(true);
// 		try {
// 			const params = new URLSearchParams({
// 				page: String(currentPage),
// 				pageSize: String(PAGE_SIZE),
// 			});

// 			if (term) {
// 				if (mode === "tag") {
// 					params.set("tag_name", term);
// 				} else {
// 					params.set("search", term);
// 					params.set("search_by", mode === "content" ? "description" : "title");
// 				}
// 			}

// 			const res = await fetch(`/api/usecases?${params}`);
// 			const json = await res.json();

// 			if (json.success) {
// 				const mapped: CaseStudy[] = (json.data || []).map((u: any) => ({
// 					id: u.id,
// 					title: u.title,
// 					description: u.description ?? "",
// 					tags: (u.tags || []).map((t: any) => (typeof t === "string" ? t : t.name)),
// 					image: u.cover_img ?? "",
// 					category: u.category_id ? String(u.category_id) : "",
// 					htmlFile: "",
// 				}));
// 				setUsecases(mapped);
// 				setTotal(json.pagination?.total ?? 0);
// 				setTotalPages(json.pagination?.totalPages ?? 1);
// 			}
// 		} catch (e) {
// 			console.error(e);
// 		} finally {
// 			setLoading(false);
// 		}
// 	}, []);

// 	useEffect(() => {
// 		fetchUsecases(page, searchTerm, searchMode);
// 	}, [page, searchTerm, searchMode, fetchUsecases]);

// 	const handleSearch = useCallback((term: string, mode: LocalSearchMode, _cat: CATEGORY) => {
// 		setSearchTerm(term);
// 		setSearchMode(mode);
// 		setPage(1);
// 	}, []);

// 	return (
// 		<div className="flex min-h-screen flex-col bg-[#f7f9fb] text-black transition-colors duration-200 dark:bg-gray-900 dark:text-white">
// 			<Header />

// 			<main className="flex-grow">
// 				<div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-10">
// 					<section className="mb-8 rounded-[28px] border border-gray-200 bg-white px-6 py-8 shadow-sm dark:border-gray-700 dark:bg-gray-800 sm:px-8">
// 						<div className="mb-4 inline-flex items-center rounded-full bg-green-50 px-4 py-1.5 text-sm font-semibold text-green-700 dark:bg-green-900/30 dark:text-green-300">
// 							Open Data Use Cases
// 						</div>
// 						<div className="max-w-3xl">
// 							<h1 className="mb-3 text-4xl font-bold tracking-tight sm:text-5xl">
// 								Use Cases
// 							</h1>
// 						</div>
// 					</section>

// 					<SearchBar onSearch={handleSearch} />

// 					{loading ? (
// 						<div className="flex justify-center py-20">
// 							<div className="h-8 w-8 animate-spin rounded-full border-4 border-green-500 border-t-transparent" />
// 						</div>
// 					) : (
// 						<PreviewComponent
// 							caseStudies={usecases}
// 							page={page}
// 							totalPages={totalPages}
// 							total={total}
// 							onPageChange={setPage}
// 						/>
// 					)}
// 				</div>
// 			</main>

// 			<div className="fixed bottom-4 right-4 z-50">
// 				<Tooglebutton onValueChange={handleToggle} />
// 			</div>

// 			<Footer />
// 		</div>
// 	);
// };

// export default UseCases;
"use client";

import React, { Suspense, useState, useEffect, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import SearchBar, { LocalSearchMode } from "./searchbar";
import PreviewComponent from "./preview";
import { CATEGORY, CaseStudy } from "../../types";
import Tooglebutton from "../Tooglebutton/Tooglebutton";

const PAGE_SIZE = 9;

const UseCases: React.FC = () => {
	const searchParams = useSearchParams();
	const [darkMode, setDarkMode] = useState(false);
	const [usecases, setUsecases] = useState<CaseStudy[]>([]);
	const [loading, setLoading] = useState(true);
	const [page, setPage] = useState(1);
	const [totalPages, setTotalPages] = useState(1);
	const [total, setTotal] = useState(0);

	const initTerm = searchParams.get("tag_name") ?? searchParams.get("search") ?? "";
	const initMode: LocalSearchMode = searchParams.get("tag_name")
		? "tag"
		: (searchParams.get("search_by") as LocalSearchMode | null) ?? "title";

	const [searchTerm, setSearchTerm] = useState<string>(initTerm);
	const [searchMode, setSearchMode] = useState<LocalSearchMode>(initMode);
	const [debouncedTerm, setDebouncedTerm] = useState<string>(initTerm);
	const [debouncedMode, setDebouncedMode] = useState<LocalSearchMode>(initMode);

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

	const fetchUsecases = useCallback(
		async (currentPage: number, term: string, mode: LocalSearchMode) => {
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
						params.set("search_by", mode);
					}
				}

				const res = await fetch(`/api/usecases?${params}`);
				const json = await res.json();

				if (json.success) {
					const mapped: CaseStudy[] = (json.data || []).map((u: any) => ({
						id: u.id,
						title: u.title,
						description: u.description ?? "",
						tags: (u.tags || []).map((t: any) =>
							typeof t === "string" ? t : t.name
						),
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
		},
		[]
	);

	// Debounce: update fetch trigger 350ms after typing stops
	useEffect(() => {
		const timer = setTimeout(() => {
			setDebouncedTerm(searchTerm);
			setDebouncedMode(searchMode);
			setPage(1);
		}, 350);
		return () => clearTimeout(timer);
	}, [searchTerm, searchMode]);

	useEffect(() => {
		fetchUsecases(page, debouncedTerm, debouncedMode);
	}, [page, debouncedTerm, debouncedMode, fetchUsecases]);

	// Submit: bypass debounce, fire immediately
	const handleSearch = useCallback(
		(term: string, mode: LocalSearchMode, _cat: CATEGORY) => {
			setSearchTerm(term);
			setSearchMode(mode);
			setDebouncedTerm(term);
			setDebouncedMode(mode);
			setPage(1);
		},
		[]
	);

	// Live typing callback from SearchBar
	const handleTermChange = useCallback((term: string, mode: LocalSearchMode) => {
		setSearchTerm(term);
		setSearchMode(mode);
	}, []);

	return (
		<div className="flex min-h-screen flex-col bg-[#f7f9fb] text-black transition-colors duration-200 dark:bg-gray-900 dark:text-white">
			<Header />

			<main className="flex-grow">
				<section className="relative overflow-hidden bg-gradient-to-br from-green-700 via-green-600 to-green-500 px-4 pb-20 pt-20 text-center dark:from-green-900 dark:via-green-800 dark:to-green-700">
					<div className="pointer-events-none absolute -left-16 -top-16 h-64 w-64 rounded-full bg-white/10 blur-2xl" />
					<div className="pointer-events-none absolute -bottom-12 -right-12 h-80 w-80 rounded-full bg-white/10 blur-2xl" />

					<div className="relative mx-auto max-w-3xl">
						<span className="mb-4 inline-block rounded-full bg-white/20 px-4 py-1 text-xs font-semibold uppercase tracking-widest text-green-100">
							Melbourne Open Data
						</span>

						<h1
							className="mb-5 text-5xl text-white drop-shadow-sm sm:text-6xl"
							style={{
								fontWeight: 900,
								fontStyle: "normal",
								fontFamily: "'Barlow Condensed', sans-serif",
								letterSpacing: "-0.02em",
								textShadow: "2px 2px 8px rgba(0,0,0,0.35)",
							}}
						>
							Use Cases
						</h1>

						<p className="mx-auto max-w-2xl text-lg leading-relaxed text-green-100">
							Explore real examples of how Melbourne open data can be used to
							understand communities, improve services, and support smarter city
							decisions.
						</p>
					</div>

					<div className="absolute bottom-0 left-0 right-0 overflow-hidden leading-none">
						<svg
							viewBox="0 0 1440 40"
							xmlns="http://www.w3.org/2000/svg"
							className="block w-full fill-[#f7f9fb] dark:fill-gray-900"
						>
							<path d="M0,20 C360,40 1080,0 1440,20 L1440,40 L0,40 Z" />
						</svg>
					</div>
				</section>

				<div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-10">
					<SearchBar
						onSearch={handleSearch}
						onTermChange={handleTermChange}
						initialTerm={initTerm}
						initialMode={initMode}
					/>

					{loading ? (
						<div className="flex justify-center py-20">
							<div className="h-10 w-10 animate-spin rounded-full border-4 border-green-200 border-t-green-600" />
						</div>
					) : usecases.length === 0 ? (
						<div className="flex min-h-[250px] items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-gray-50 px-6 py-12 text-center dark:border-gray-700 dark:bg-gray-800">
							<p className="text-base font-medium text-gray-500 dark:text-gray-400">
								No use cases available at the moment.
							</p>
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

export default function UseCasesPage() {
	return (
		<Suspense>
			<UseCases />
		</Suspense>
	);
}