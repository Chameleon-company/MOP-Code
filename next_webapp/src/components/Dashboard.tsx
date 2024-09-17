"use client"
import { Link } from "@/i18n-navigation";
import Image from "next/image";
import mainimage from "../../public/img/mainImage.png";
import secondimage from "../../public/img/second_image.png";
import { useTranslations } from "next-intl";
import { CaseStudy, CATEGORY, SEARCH_MODE, SearchParams } from "@/app/types";
import { useEffect, useState } from "react";
import { ArrowLeft, FileText } from "lucide-react";


const Dashboard = () => {
    const [filteredCaseStudies, setFilteredCaseStudies] = useState<CaseStudy[]>([]);
    const [selectedCaseStudy, setSelectedCaseStudy] = useState<CaseStudy | undefined>(undefined)

    useEffect(() => {
        handleSearch("", SEARCH_MODE.TITLE, CATEGORY.ALL)
    }, [])

    async function searchUseCases(searchParams: SearchParams) {
        const response = await fetch("/api/search-use-cases", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(searchParams),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }
    const handleSearch = async (
        searchTerm: string,
        searchMode: SEARCH_MODE,
        category: CATEGORY
    ) => {
        const res = await searchUseCases({ searchTerm, searchMode, category });
        console.log("🚀 ~ UseCases ~ res:", res);
        setFilteredCaseStudies(res?.filteredStudies);
    };
    const handleCaseStudyClick = (study: CaseStudy) => {
        setSelectedCaseStudy(study);
    };
    const handleBack = () => {
        setSelectedCaseStudy(undefined);
    };
    const navItems = [
        { to: "/about", icon: "/img/about-icon.png", label: "About Us" },
        { to: "/casestudies", icon: "/img/case-icon.png", label: "Case Studies" },
        {
            to: "/resource-center",
            icon: "/img/resource-icon.png",
            label: "Resource Center",
        },
        { to: "/datasets", icon: "/img/data-icon.png", label: "Data Collection" },
        { to: "/contact", icon: "/img/contact-icon.png", label: "Contact Us" },
    ];

    const t = useTranslations("common");

    if (selectedCaseStudy) {
        return (
            <div className="flex flex-col h-screen bg-gray-100 p-8">
                <button
                    onClick={handleBack}
                    className="flex items-center text-green-500 mb-4 hover:text-green-700 transition-colors duration-300"
                >
                    <ArrowLeft size={24} className="mr-2" />
                    Back
                </button>
                <div className="bg-white rounded-lg shadow-md p-6 flex-grow overflow-hidden">
                    <h1 className="text-3xl font-bold mb-4">{selectedCaseStudy.name}</h1>
                    <iframe
                        src={`/api?filename=${selectedCaseStudy.filename}`}
                        className="w-full h-full border-none"
                        title={selectedCaseStudy.name}
                    />
                </div>
            </div>
        );
    }

    return (
        <>
            <style dangerouslySetInnerHTML={{ __html: style }} />

            <div className="main-wrapper">
                <div className="main-container">
                    <section className="hero-section">
                        <Image src={mainimage} alt={"main image1"} />
                    </section>
                    <section className="sign-up-btn-section">
                        <button className="sign-up-btn">
                            <Link href="signup">{t("Sign Up")}</Link>
                        </button>
                    </section>
                    <section className="our-vision-section">
                        <div className="our-vision">{t("Our Vision")}</div>
                        <div className="img-div">
                            <Image src={secondimage} alt={"Second Image"} />
                        </div>
                        <div className="text-div">{t("intro")}</div>
                    </section>
                    <section className="recent-case-studies">
                        <h2>{t("Recent Case Studies")}</h2>
                        <p>{t("p2")}</p>
                    </section>

                    <section className="case-studies mx-10">
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {filteredCaseStudies.slice(0, 3).map((study) => (
                                <div
                                    key={study.id}
                                    className="bg-white p-4 rounded-lg border-4 shadow-md cursor-pointer hover:shadow-2xl transition-shadow duration-300"
                                    onClick={() => handleCaseStudyClick(study)}
                                >
                                    <div className="flex items-center justify-center mb-4">
                                        <FileText size={48} className="text-green-500" />
                                        <FileText size={48} className="text-teal-400 -ml-6" />
                                        <FileText size={48} className="text-green-700 -ml-6 rotate-6" />
                                    </div>
                                    <h3 className="font-bold text-lg text-center mb-2">{study.name}</h3>
                                    <p className="text-gray-600 text-sm text-center mb-2">{study.description}</p>
                                    <div className="flex flex-wrap justify-center gap-2">
                                        <p className="text-sm text-gray-500">Tags:</p>
                                        {study.tags.map((tag, index) => (
                                            <span
                                                key={index}
                                                className="bg-gray-200 text-gray-800 text-xs px-2 py-1 rounded-full"
                                            >
                                                {tag}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>

                    </section>
                </div>
            </div>
        </>
    );
};

export default Dashboard;