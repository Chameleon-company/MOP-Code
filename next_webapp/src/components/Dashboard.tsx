"use client";
import { Link } from "@/i18n-navigation";
import Image from "next/image";
import mainimage from "../../public/img/mainImage.png";
import secondimage from "../../public/img/second_image.png";
import { useTranslations } from "next-intl";
import { CaseStudy, CATEGORY, SEARCH_MODE, SearchParams } from "@/app/types";
import { useEffect, useState, useRef } from "react";
import {
	ArrowLeft,
	FileText,
	ArrowRight,
	Play,
	ChevronDown,
	Search,
	X,
} from "lucide-react";
import Carousel from "react-multi-carousel";
import "react-multi-carousel/lib/styles.css";

import { Users, Car, Trees, Home, DollarSign, Heart } from "lucide-react";

import HeroCarousel from "./HeroCarousel";



const style = `
.main-container {
  background: white;
  color: #263238;
  display: flex;
  flex-direction: column;
  margin-bottom: 0;
}
.dark .main-container {
  background: #263238;
  color: white;
  flex: 1;
}

/* Enhanced Hero Section */
.hero-section {
  width: 100%;
  height: 90vh;
  min-height: 700px;
  overflow: hidden;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}
.hero-image-container {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 1;
}
.hero-image-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(to right, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.3) 50%, rgba(0,0,0,0.7) 100%);
  z-index: 2;
}
.dark .hero-image-container::before {
  background: linear-gradient(to right, rgba(38,50,56,0.8) 0%, rgba(38,50,56,0.4) 50%, rgba(38,50,56,0.8) 100%);
}
.hero-image-container img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.hero-content {
  position: relative;
  z-index: 3;
  text-align: center;
  color: white;
  max-width: 900px;
  padding: 0 2rem;
  animation: fadeInUp 1s ease-out;
}
.hero-title {
  font-size: 3.5rem;
  font-weight: 800;
  margin-bottom: 1.5rem;
  line-height: 1.2;
  text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
.hero-subtitle {
  font-size: 1.5rem;
  margin-bottom: 2.5rem;
  font-weight: 400;
  max-width: 700px;
  margin-left: auto;
  margin-right: auto;
}
.hero-buttons {
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin-bottom: 2rem;
}
.hero-button {
  padding: 1rem 2rem;
  border-radius: 50px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.3s ease;
}
.hero-button.primary {
  background: #10B981;
  color: white;
}
.hero-button.primary:hover {
  background: #059669;
  transform: translateY(-2px);
  box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
}
.hero-button.secondary {
  background: transparent;
  color: white;
  border: 2px solid white;
}
.hero-button.secondary:hover {
  background: rgba(255,255,255,0.1);
  transform: translateY(-2px);
}

/* Search Container */
.search-container {
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(10px);
  border-radius: 50px;
  padding: 0.5rem;
  margin: 0 auto;
  max-width: 600px;
  display: flex;
  align-items: center;
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);

  z-index: 10;
}
.search-input {
  flex: 1;
  background: transparent;
  border: none;
  padding: 0.8rem 1.5rem;
  color: white;
  font-size: 1rem;
  outline: none;
}
.search-input::placeholder {
  color: rgba(255, 255, 255, 0.7);
}
.search-options {
  display: flex;
  gap: 0.5rem;
  padding: 0 1rem;
 
}
.search-option {
  background: rgba(255, 255, 255, 0.2);
  border: none;
  border-radius: 20px;
  padding: 0.4rem 0.8rem;
  color: white;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.3s ease;
}
.search-option:hover, .search-option.active {
  background: #10B981;
}
.search-button {
  background: #10B981;
  border: none;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
}
.search-button:hover {
  background: #059669;
  transform: rotate(15deg);
}
.clear-search {
  background: rgba(255, 255, 255, 0.2);
  border: none;
  border-radius: 50%;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  margin-right: 0.5rem;
  transition: all 0.3s ease;
}
.clear-search:hover {
  background: rgba(255, 255, 255, 0.3);
}

.search-results {
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  width: 90%;
  max-width: 800px;
  background: white;
  border-radius: 12px;
  margin-top: 1rem;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
  max-height: 400px;
  overflow-y: auto;
  z-index: 10;
}
.dark .search-results {
  background: #37474f;
  color: white;
}
.search-result-item {
  padding: 1rem;
  border-bottom: 1px solid #eee;
  cursor: pointer;
  transition: all 0.2s ease;
}
.dark .search-result-item {
  border-bottom: 1px solid #546e7a;
}
.search-result-item:hover {
  background: #f5f5f5;
}
.dark .search-result-item:hover {
  background: #455a64;
}
.search-result-item h4 {
  margin: 0 0 0.5rem 0;
  color: #263238;
}
.dark .search-result-item h4 {
  color: white;
}
.search-result-item p {
  margin: 0;
  font-size: 0.9rem;
  color: #546e7a;
}
.dark .search-result-item p {
  color: #b0bec5;
}
.no-results {
  padding: 1.5rem;
  text-align: center;
  color: #78909c;
}
.dark .no-results {
  color: #b0bec5;
}

.scroll-indicator {
  position: absolute;
  bottom: 30px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 3;
  animation: bounce 2s infinite;
  color: white;
}
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
@keyframes bounce {
  0%, 20%, 50%, 80%, 100% {
    transform: translateY(0) translateX(-50%);
  }
  40% {
    transform: translateY(-20px) translateX(-50%);
  }
  60% {
    transform: translateY(-10px) translateX(-50%);
  }
}
@media (max-width: 768px) {
  .hero-title {
    font-size: 2.5rem;
  }
  .hero-subtitle {
    font-size: 1.2rem;
  }
  .hero-buttons {
    flex-direction: column;
    align-items: center;
  }
  .search-container {
    flex-direction: column;
    border-radius: 12px;
    padding: 1rem;
  }
  .search-input {
    width: 100%;
    margin-bottom: 0.5rem;
  }
  .search-options {
    width: 100%;
    justify-content: center;
    margin-bottom: 0.5rem;
  }
  .search-option {
    flex: 1;
    text-align: center;
  }
}

.our-vision-section {
  display: flex;
  flex-direction: column;
  background: white;
  color: #263238;
  margin: 3rem auto;
  gap: 2rem;
    top: 100%;  /* Position below the search bar */
  padding: 2rem;
  width: 90%;
  max-width: 1200px;
  box-sizing: border-box;
}
.dark .our-vision-section {
  background: #263238;
  color: white;
}
@media (min-width: 768px) {
  .our-vision-section {
    flex-direction: row;
    height: auto;
    min-height: 10rem;
    
  }
}
.img-div {
  width: 100%;
  height: 100%;
}
.img-div img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.text-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.our-vision {
  font-weight: 900;
  font-size: 2rem;
}
.case-studies-wrapper {
  background-color: #F6F9FC;
  padding: 2rem;
}
.dark .case-studies-wrapper {
  background-color: rgb(46, 38, 38);
}
.recent-case-studies {
  margin: 0 auto;
  padding: 0 2rem;
  max-width: 1200px;
  text-align: center;
  color: #263238;
}
.dark .recent-case-studies {
  color: white;
}
.recent-case-studies h2 {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 2rem;
}
.recent-case-studies p {
  font-size: 1rem;
  margin-bottom: 3rem;
  width: 50%;
  margin-left: auto;
  margin-right: auto;
}
.case-studies {
  margin: 2rem auto 0 auto;
  padding: 0 1rem;
  max-width: 1400px;
  color: #263238;
}
.dark .case-studies {
  color: white;
}
.case-study {
  background: white;
  color: #263238;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  transition: box-shadow 0.3s;
  margin: 0 0.5rem;
  height: 80%;
  border-radius: 0.5rem;
}
.dark .case-study {
  background: #37474f;
  color: white;
}
.case-study:hover {
  box-shadow: 0 8px 12px rgba(0,0,0,0.2);
}
.case-study img {
  width: 100%;
  height: auto;
  object-fit: cover;
}
.react-multi-carousel-item {
  padding: 0 0.5rem;
}

/* Loading states */
.loading-spinner {
  border: 3px solid #f3f3f3;
  border-top: 3px solid #10B981;
  border-radius: 50%;
  width: 30px;
  height: 30px;
  animation: spin 1s linear infinite;
  margin: 0 auto;
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Debug info panel */
.debug-panel {
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  padding: 1rem;
  margin: 1rem 0;
  font-family: monospace;
  font-size: 0.8rem;
  max-height: 200px;
  overflow-y: auto;
}
.dark .debug-panel {
  background: #2d3748;
  border-color: #4a5568;
}
`;

const categories = [
  { icon: "🏢", label: "EV Infrastructure" },
  { icon: "🅿️", label: "Parking" },
  { icon: "🚨", label: "Safety" },
];

const openPage = (label: string) => {
  if (label === "EV Infrastructure") {
    window.location.href = "/en/ev-infrastructure";
  }
};

const responsive = {
	superLargeDesktop: { breakpoint: { max: 4000, min: 3000 }, items: 5 },
	desktop: { breakpoint: { max: 3000, min: 1024 }, items: 3 },
	tablet: { breakpoint: { max: 1024, min: 464 }, items: 2 },
	mobile: { breakpoint: { max: 464, min: 0 }, items: 1 },
};

// Get available search modes from the SEARCH_MODE enum
const getSearchModeValues = () => {
	try {
		const searchModes = Object.values(SEARCH_MODE).filter(
			(value) => typeof value === "string"
		) as string[];

		if (searchModes.length === 0) {
			return ["TITLE", "CONTENT"];
		}

		return searchModes;
	} catch (error) {
		return ["TITLE", "CONTENT"];
	}
};

const Dashboard = () => {
	const t = useTranslations("common");
	const t_hero = useTranslations("hero");
	const [filteredCaseStudies, setFilteredCaseStudies] = useState<CaseStudy[]>(
		[]
	);
	const [selectedCaseStudy, setSelectedCaseStudy] = useState<
		CaseStudy | undefined
	>(undefined);
	const [searchTerm, setSearchTerm] = useState("");
	const [searchMode, setSearchMode] = useState<SEARCH_MODE>(SEARCH_MODE.TITLE);
	const [category, setCategory] = useState<CATEGORY>(CATEGORY.ALL);
	const [showSearchResults, setShowSearchResults] = useState(false);
	const [isSearching, setIsSearching] = useState(false);
	const [debugInfo, setDebugInfo] = useState<any>(null);

	// Create ref for the search container
	const searchContainerRef = useRef<HTMLDivElement>(null);

	// Get available search modes
	const availableSearchModes = getSearchModeValues();

	useEffect(() => {
		handleSearch("", SEARCH_MODE.TITLE, CATEGORY.ALL);
	}, []);

	// Add click outside handler
	useEffect(() => {
		const handleClickOutside = (event: MouseEvent) => {
			if (
				searchContainerRef.current &&
				!searchContainerRef.current.contains(event.target as Node)
			) {
				setShowSearchResults(false);
			}
		};

		document.addEventListener("mousedown", handleClickOutside);
		return () => {
			document.removeEventListener("mousedown", handleClickOutside);
		};
	}, []);

	const searchUseCases = async (searchParams: SearchParams) => {
		const response = await fetch("/api/search-use-cases", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(searchParams),
		});

		if (!response.ok) {
			throw new Error('HTTP error! status: ${response.status}');
		}

		return await response.json();
	};

	const handleSearch = async (
		searchTerm: string,
		searchMode: SEARCH_MODE,
		category: CATEGORY
	) => {
		setIsSearching(true);
		try {
			const res = await searchUseCases({ searchTerm, searchMode, category });
			setFilteredCaseStudies(res?.filteredStudies || []);
			setDebugInfo(res?.debug || null);
		} catch (error) {
			console.error("Search error:", error);
		} finally {
			setIsSearching(false);
		}
	};

	const handleCaseStudyClick = (study: CaseStudy) => {
		setSelectedCaseStudy(study);
		setShowSearchResults(false);
	};

	const handleBack = () => {
		setSelectedCaseStudy(undefined);
	};

	const scrollToContent = () => {
		document.querySelector(".our-vision-section")?.scrollIntoView({
			behavior: "smooth",
		});
	};

	const handleSearchSubmit = (e: React.FormEvent) => {
		e.preventDefault();
		handleSearch(searchTerm, searchMode, category);
		setShowSearchResults(true);
	};

	const clearSearch = () => {
		setSearchTerm("");
		setShowSearchResults(false);
		handleSearch("", SEARCH_MODE.TITLE, CATEGORY.ALL);
	};

	if (selectedCaseStudy) {
		return (
			<div className="flex flex-col h-screen bg-gray-100 dark:bg-gray-900 text-black dark:text-white p-8">
				<button
					onClick={handleBack}
					className="flex items-center text-green-500 mb-4 hover:text-green-700 transition-colors duration-300"
				>
					<ArrowLeft size={24} className="mr-2" />
					Back
				</button>
				<div className="bg-white dark:bg-gray-800 rounded-lg shadow-md flex-grow overflow-hidden flex flex-col">
					<h1 className="text-3xl font-bold mb-4 px-6 pt-6">
						{selectedCaseStudy.name}
					</h1>
					<iframe
						src={'/api?filename=${selectedCaseStudy.filename}'}
						title={selectedCaseStudy.name}
						className="flex-grow w-full border-none bg-white dark:bg-gray-900 text-black dark:text-white"
					/>
				</div>
			</div>
		);
	}

	return (
		<>
			<style dangerouslySetInnerHTML={{ __html: style }} />
			<div className="main-wrapper bg-white dark:bg-[#263238] text-black dark:text-white min-h-screen">
				<div className="main-container">
					<section className="hero-section">
						<div className="hero-image-container">
							<Image
								src={mainimage}
								alt="main image"
								priority
								placeholder="blur"
							/>
						</div>

						<div className="hero-content">
							<h1 className="hero-title">{t_hero("hero-top")}</h1>
							<p className="hero-subtitle">{t_hero("hero-sub")}</p>

							<div className="hero-buttons">
								<button className="hero-button primary">
									{t_hero("exploreCaseStudies")} <ArrowRight size={20} />
								</button>
								<a
									href="https://youtu.be/D-H-nCrQDZo"
									target="_blank"
									rel="noopener noreferrer"
									className="hero-button secondary inline-flex items-center gap-2"
								>
									<Play size={20} fill="white" />
									{t_hero("watchVideo")}
								</a>
							</div>

							{/* Search Container with ref for outside click detection */}
							<div
								ref={searchContainerRef}
								style={{ position: "relative", width: "100%" }}
							>
								<form
									onSubmit={handleSearchSubmit}
									className="search-container"
								>
									<input
										type="text"
										placeholder={t_hero("place_holder")}
										className="search-input"
										value={searchTerm}
										onChange={(e) => setSearchTerm(e.target.value)}
										onFocus={() => setShowSearchResults(true)}
									/>
									{searchTerm && (
										<button
											type="button"
											className="clear-search"
											onClick={clearSearch}
										>
											<X size={16} />
										</button>
									)}
									<div className="search-options">
										{availableSearchModes.map((mode) => (
											<button
												key={mode}
												type="button"
												className={`search-option ${
													searchMode === mode ? "active" : ""
												}`}
												onClick={() => setSearchMode(mode as SEARCH_MODE)}
											>
												{mode.charAt(0) + mode.slice(1).toLowerCase()}
											</button>
										))}
									</div>
									<button type="submit" className="search-button">
										<Search size={20} />
									</button>
								</form>

								{showSearchResults && (
									<div className="search-results">
										{isSearching ? (
											<div className="no-results">
												<div className="loading-spinner"></div>
												<p>Searching...</p>
											</div>
										) : filteredCaseStudies.length > 0 ? (
											filteredCaseStudies.map((study) => (
												<div
													key={study.id}
													className="search-result-item"
													onClick={() => handleCaseStudyClick(study)}
												>
													<h4>{study.name}</h4>
													<p>
														{study.description.split(" ").length > 30
															? `${study.description
																	.split(" ")
																	.slice(0, 30)
																	.join(" ")}...`
															: study.description}
													</p>
												</div>
											))
										) : (
											<div className="no-results">
												No case studies found. Try a different search.
											</div>
										)}
									</div>
								)}
							</div>
						</div>

						<div className="scroll-indicator" onClick={scrollToContent}>
							<ChevronDown size={40} />
						</div>
					</section>
					<section className="our-vision-section">
						<div className="img-div">
							<Image src={secondimage} alt="Second Image" />
						</div>
						<div className="text-container">
							<h2 className="our-vision">{t("Our Vision")}</h2>
							<p>{t("intro")}</p>
						</div>
					</section>

					<section className="case-studies-wrapper">
						<section className="recent-case-studies">
							<h2>{t("Recent Case Studies")}</h2>
							<p>{t("p2")}</p>
						</section>


						<section className="case-studies">
							<Carousel responsive={responsive}>
								{filteredCaseStudies.slice(0, 6).map((study) => (
									<div
										key={study.id}
										className="case-study p-4 cursor-pointer hover:shadow-2xl transition-shadow duration-300"
										onClick={() => handleCaseStudyClick(study)}
									>
										<div className="flex items-center justify-center mb-4">
											<FileText size={48} className="text-green-500" />
											<FileText size={48} className="text-teal-400 -ml-6" />
											<FileText
												size={48}
												className="text-green-700 -ml-6 rotate-6"
											/>
										</div>
										<h3 className="font-bold text-lg text-center mb-2">
											{study.name}
										</h3>
										<p className="text-gray-600 dark:text-gray-300 text-sm text-center mb-2">
											{study.description.split(" ").length > 50
												? `${study.description
														.split(" ")
														.slice(0, 50)
														.join(" ")}...`
												: study.description}
										</p>
										<div className="flex flex-wrap justify-center gap-2">
											<p className="text-sm text-gray-500 dark:text-gray-400">
												Tags:
											</p>
											{study.tags &&
												study.tags.map((tag, index) => (
													<span
														key={index}
														className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 text-xs px-2 py-1 rounded-full"
													>
														{tag}
													</span>
												))}
										</div>
									</div>
								))}
							</Carousel>
						</section>
					</section>
				</div>
			</div>
		</>
	);

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: style }} />
      <div className="main-wrapper bg-white dark:bg-[#263238] text-black dark:text-white min-h-screen">
        <div className="main-container">
          <section className="hero-section">
             {/*<Image src={mainimage} alt="main image1" />*/}
            <HeroCarousel />
          </section>

          <section className="our-vision-section">
            <div className="img-div">
              <Image src={secondimage} alt="Second Image" />
            </div>
            <div className="text-container">
              <h2 className="our-vision">{t("Our Vision")}</h2>
              <p>{t("intro")}</p>
            </div>
          </section>

          <section className="w-full max-w-6xl mx-auto mt-10 px-2 md:px-0">
            <h3 className="text-xl font-semibold mb-3">Explore by category</h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-4 mb-2">
                  {categories.map((cat, idx) => (
                <div
                  key={idx}
                  onClick={() => openPage(cat.label)}
                  className="rounded-2xl border shadow-sm py-6 flex flex-col items-center bg-white dark:bg-gray-800 hover:shadow-md transition cursor-pointer"
                >
                  <span className="text-3xl mb-2">{cat.icon}</span>
                  <span className="text-sm font-medium text-center">{cat.label}</span>
                </div>
                ))}
              </div>
          </section>

          <section className="case-studies-wrapper">
            <section className="recent-case-studies">
              <h2>{t("Recent Case Studies")}</h2>
              <p>{t("p2")}</p>
            </section>

            <section className="case-studies">
              <Carousel responsive={responsive}>
                {filteredCaseStudies.slice(0, 6).map((study) => (
                  <div
                    key={study.id}
                    className="case-study p-4 cursor-pointer hover:shadow-2xl transition-shadow duration-300"
                    onClick={() => handleCaseStudyClick(study)}
                  >
                    <div className="flex items-center justify-center mb-4">
                      <FileText size={48} className="text-green-500" />
                      <FileText size={48} className="text-teal-400 -ml-6" />
                      <FileText size={48} className="text-green-700 -ml-6 rotate-6" />
                    </div>
                    <h3 className="font-bold text-lg text-center mb-2">{study.name}</h3>
                    <p className="text-gray-600 dark:text-gray-300 text-sm text-center mb-2">
                      {study.description.split(" ").length > 50
                        ? `${study.description.split(" ").slice(0, 50).join(" ")}...`
                        : study.description}
                    </p>
                    <div className="flex flex-wrap justify-center gap-2">
                      <p className="text-sm text-gray-500 dark:text-gray-400">Tags:</p>
                      {study.tags.map((tag, index) => (
                        <span
                          key={index}
                          className="bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 text-xs px-2 py-1 rounded-full"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </Carousel>
            </section>
          </section>
        </div>
      </div>
    </>
  );

};

export default Dashboard;
