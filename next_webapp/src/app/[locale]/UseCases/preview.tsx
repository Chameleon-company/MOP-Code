import { useEffect, useState } from "react";
import { CaseStudy } from "../types";
import CaseStudyDetail from "./casestudydetail";

const PreviewComponent = ({ caseStudies }: { caseStudies: CaseStudy[] }) => {
  // State to keep track of the selected case study
  const [selectedCaseStudy, setSelectedCaseStudy] = useState<CaseStudy | undefined>(caseStudies[0]); // Default to the first case study
const [clicked,setclicked]=useState(false)
  useEffect(() => {
    setSelectedCaseStudy(caseStudies[0]);
  }, [caseStudies]);

  return (
    <div className="flex h-screen">
      {/* Scrollable Menu on the left */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gridAutoRows: "20%", width: "100%" }}>
        <ul style={{ display: 'contents' }}>
          {caseStudies.map((study) => (
          //  <a href="/en/detailed">
             <li
              key={study.id}
              className={`text-black mb-2 p-2 hover:bg-gray-200 cursor-pointer ${
                selectedCaseStudy?.id === study.id ? "bg-gray-300" : ""
              }`}
              onClick={() =>{ setSelectedCaseStudy(study),setclicked(!clicked)}}
            >
              {study.title}
            </li>
          //  </a>
          ))}
        </ul>
      </div>

      {/* Preview Screen on the right */}
      {clicked && (
        <div className="flex-grow">
          <CaseStudyDetail selectedCaseStudy={selectedCaseStudy} />
        </div>
      )}
    </div>
  );
};

export default PreviewComponent;
