// DashboardCaseStd.tsx

"use client"
import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useTranslations } from "next-intl";
import { CaseStudy } from "../app/types"; 
import { FileText } from "lucide-react"; 

const DashboardCaseStd = () => {
  const [caseStudies, setCaseStudies] = useState<CaseStudy[]>([]);
  const t = useTranslations("common");

  useEffect(() => {
    const fetchCaseStudies = async () => {
      try {
        const response = await fetch('/api/usecases');
        const data = await response.json();
        console.log("Fetched Data:", data); // Check the data length and content
        setCaseStudies(data.slice(0, 3)); // Fetch only the first 3 case studies
      } catch (error) {
        console.error("Error fetching case studies:", error);
      }
    };
  
    fetchCaseStudies();
  }, []);
  
  

  if (caseStudies.length === 0) {
    return <p>Loading case studies...</p>; // Shows loading text if data isn't fetched yet
  }

  return (
    
    <div className="case-studies-wrapper">
      {caseStudies.map((caseStudy, index) => (
        <Link href={`en/UseCases`} key={index}>  {}
          <div className="bg-white p-8 rounded-lg shadow-md cursor-pointer hover:shadow-lg transition-shadow duration-300">
            <div className="flex items-center justify-center mb-4">
              <FileText size={48} className="text-green-500" />
              <FileText size={48} className="text-teal-400 -ml-6" />
              <FileText size={48} className="text-green-700 -ml-6 rotate-6" />
            </div>
            <h3 className="font-bold text-lg text-center mb-2">{caseStudy.name}</h3>
            <p className="text-gray-600 text-center mb-2">{caseStudy.description}</p>
            <div className="flex flex-wrap justify-center gap-2">
              <p>Tags: </p>
              {caseStudy.tags.map((tag, index) => (
                <span
                  key={index}
                  className="bg-gray-200 text-gray-800 text-sm px-2 py-1 rounded-full"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        </Link>
      ))}
    </div>
  );
};
export default DashboardCaseStd;
