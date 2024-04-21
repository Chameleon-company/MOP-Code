import { useState } from 'react';
import { database } from '../UseCases/database'; 

const PreviewComponent = () => {
  // State to keep track of the selected case study
  const [selectedCaseStudy, setSelectedCaseStudy] = useState(database[0]); // Default to the first case study

  return (
    <div className="flex h-screen">
      {/* Scrollable Menu on the left */}
      <div className="w-1/4 overflow-y-auto bg-gray-100 p-4">
        <ul>
          {database.map((study) => (
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

      {/* Preview Screen */}
      <div className="w-3/4 bg-gray-200 p-4 overflow-y-auto">
        <div className="h-full w-full">
          {/* Displaying the PDF */}
          {selectedCaseStudy && (
                <div style={{ width: "100%" }}>
                <iframe
                  style={{ width: "100%", height: "100vh" }}
                  src={selectedCaseStudy.caseUrl}
                ></iframe>
              </div>
            // <object data={selectedCaseStudy.caseUrl} type="application/pdf" width="100%" height="100%">
            //   <p>Your browser does not support PDFs. <a href={selectedCaseStudy.pdf}>Download the PDF</a>.</p>
            // </object>
          )}
        </div>
      </div>
    </div>
  );
};

export default PreviewComponent;



  
  