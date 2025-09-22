"use client";
import React from "react";
import { useTranslations } from "next-intl";

export type UseCase = {
    id: string | number;
    title: string;
};

type Props = {
    featured?: UseCase[];
    allCases?: UseCase[];
};

const defaultFeatured: UseCase[] = [
    { id: 1, title: "Use Case 1" },
    { id: 2, title: "Use Case 2" },
    { id: 3, title: "Use Case 3" },
    { id: 4, title: "Use Case 4" },
];

const defaultAll: UseCase[] = [
    { id: 5, title: "Use Case 5" },
    { id: 6, title: "Use Case 6" },
    { id: 7, title: "Use Case 7" },
    { id: 8, title: "Use Case 8" },
];

const UseCaseInsights: React.FC<Props> = ({
    featured = defaultFeatured,
    allCases = defaultAll,
}) => {
    
    const [expanded, setExpanded] = React.useState(false);

    return (
        <section className="w-full bg-[#2ecc71] py-12 lg:py-16">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                {/* heading */}
                <h2 className="text-4xl sm:text-5xl font-extrabold text-center text-black mb-12">
                    Use Case Insights
                </h2>

        
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-8 mb-8">
                    {featured.map((uc) => (
                        <div
                            key={uc.id}
                            className="bg-white rounded-xl shadow-md flex items-center justify-center h-64 text-lg font-semibold text-black"
                        >
                            {uc.title}
                        </div>
                    ))}
                </div>

                {/* expand area */}
                {expanded && (
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-8 mb-8">
                        {allCases.map((uc) => (
                            <div
                                key={uc.id}
                                className="bg-white rounded-xl shadow-md flex items-center justify-center h-64 text-lg font-semibold text-black"
                            >
                                {uc.title}
                            </div>
                        ))}
                    </div>
                )}

                {/* button for expand */}
                <div className="flex justify-center">
                    <button
                        onClick={() => setExpanded(!expanded)}
                        className="px-6 py-3 bg-white rounded-full shadow-md font-bold text-black hover:shadow-lg transition"
                    >
                        {expanded ? "Hide Use Cases" : "View All Use Cases"}
                    </button>
                </div>
            </div>
        </section>
    );
};

export default UseCaseInsights;
