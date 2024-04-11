import { useState } from 'react';
import caseStudies from './database';
// Mock data for case studies with associated PDF paths


const PreviewComponent = () => {
  // State to keep track of the selected case study
  const [selectedCaseStudy, setSelectedCaseStudy] = useState(caseStudies[0]); // Default to the first case study

  return (
    <div className="flex h-screen">
      {/* Scrollable Menu on the left */}
      <div className="w-1/4 overflow-y-auto bg-gray-100 p-4">
        <ul>
          {caseStudies.map((study) => (
            <li
              key={study.id}
              className={`text-black mb-2 p-2 hover:bg-gray-200 cursor-pointer ${selectedCaseStudy.id === study.id ? 'bg-gray-300' : ''}`}
              onClick={() => setSelectedCaseStudy(study)}
            >
              {study.title}
            </li>
          ))}
        </ul>
      </div>

      {/* Preview Screen on the right */}
      <div className="w-3/4 bg-gray-200 p-4 overflow-y-auto">
        <div className="h-full w-full">
          {/* Display an iframe or object element to show the PDF */}
          {selectedCaseStudy && (
            <object data={selectedCaseStudy.pdf} type="application/pdf" width="100%" height="100%">
              <p>Your browser does not support PDFs. <a href={selectedCaseStudy.pdf}>Download the PDF</a>.</p>
            </object>
          )}
        </div>
      </div>
    </div>
  );
};

export default PreviewComponent;



  
  