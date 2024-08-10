import React from 'react';
import { CaseStudy } from '../types';

const CaseStudyDetail = ({ selectedCaseStudy }: { selectedCaseStudy: CaseStudy | undefined }) => {
  if (!selectedCaseStudy) {
    return null; // Render nothing if no case study is selected
  }

  return (
    <div className="fixed inset-0 bg-gray-200 p-0 overflow-hidden">
      <div className="h-full w-full">
        <div className="font-semibold text-2xl p-4">{selectedCaseStudy.title}</div>
        <div style={{ width: "100%", height: "calc(100% - 3rem)" }}>
          <iframe
            style={{ width: "100%", height: "100%" }}
            src={`/api?filename=${selectedCaseStudy.filename}`}
          ></iframe>
        </div>
      </div>
    </div>
  );
};

export default CaseStudyDetail;
