// CaseStudies.tsx
"use client"; // This is a client component 
import React, { useState } from 'react';
import { caseStudiesData } from './caseStudiesData';
import { CaseStudy } from './caseStudyTypes';
//jk
// Define the prop types for CaseStudyList
type CaseStudyListProps = {
    caseStudies: CaseStudy[];
    onCaseStudyClick: (caseStudy: CaseStudy) => void;
  };
  
  // Define the prop types for CaseStudyPreview
  type CaseStudyPreviewProps = {
    caseStudy: CaseStudy | null;
  };
  


// This component will be responsible for displaying the list of case studies
const CaseStudyList: React.FC<CaseStudyListProps> = ({ caseStudies, onCaseStudyClick }) => (
    <div className="w-1/5 overflow-auto" style={{ height: '50vh' }}> {/* Adjusted to 50vh */}
      <ul className="space-y-4">
        {caseStudies.map((caseStudy: CaseStudy) => (
          <li 
            key={caseStudy.id} 
            onClick={() => onCaseStudyClick(caseStudy)}
            className="p-4 border-2 border-gray-300 rounded-lg hover:bg-gray-100 cursor-pointer"
          >
            {caseStudy.title}
          </li>
        ))}
      </ul>
    </div>
  );
  

// This component will be responsible for displaying the details of the selected case study
const CaseStudyPreview: React.FC<CaseStudyPreviewProps> = ({ caseStudy }) => (
    <div className="w-3/4 ml-8 p-4 border-2 border-gray-300 rounded-lg" style={{ height: '100vh' }}>
    <div className="h-full bg-gray-200 rounded-lg overflow-hidden p-4">
      {caseStudy ? (
        <>
          <h2 className="text-2xl font-bold">{caseStudy.title}</h2>
          <p>{caseStudy.content}</p>
          {/* More details of the case study can be rendered here */}
        </>
      ) : (
        <p>Select a case study to view details</p>
      )}
    </div>
  </div>
);

const CaseStudies: React.FC = () => {
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedCategory, setSelectedCategory] = useState('All categories');
    const [selectedCaseStudy, setSelectedCaseStudy] = useState<CaseStudy | null>(null);
  
    // Filter the case studies based on the search term and selected category
    const filteredCaseStudies = caseStudiesData.filter(caseStudy =>
      (caseStudy.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      caseStudy.content.toLowerCase().includes(searchTerm.toLowerCase())) &&
      (selectedCategory === 'All categories' || caseStudy.group === selectedCategory)
    );
  
    // Update selectedCaseStudy when a case study from the list is clicked
    const handleCaseStudyClick = (caseStudy: CaseStudy) => {
      setSelectedCaseStudy(caseStudy);
    };

    return (
        <div className="container mx-auto px-1 py-2 pb-4">
          <h1 className="text-4xl font-bold mb-8">Case Studies</h1>
          <div className="flex flex-wrap items-center mb-8">
            <div className="relative flex-grow mr-4">
              <input 
                type="text" 
                placeholder="Case study name or category"
                className="w-full p-4 border-2 border-gray-300 rounded-l-lg"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
              <span className="absolute inset-y-0 right-0 flex items-center pr-2">
                <i className="fas fa-search text-gray-500"></i>
              </span>
            </div>
            <div className="relative">
              <select 
                className="border-2 border-gray-300 rounded-r-lg p-4"
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
              >
                <option>All categories</option>
                {/* Dynamically generate category options */}
              </select>
            </div>
            <button 
              className="bg-green-500 text-white rounded-lg p-4 ml-4 hover:bg-green-600"
              onClick={() => setSearchTerm(searchTerm)} 
            >
              Search
            </button>
          </div>
          <div className="flex">
            <CaseStudyList caseStudies={filteredCaseStudies} onCaseStudyClick={handleCaseStudyClick} />
            <CaseStudyPreview caseStudy={selectedCaseStudy} />
          </div>
        </div>
      );
    };
    
    export default CaseStudies;