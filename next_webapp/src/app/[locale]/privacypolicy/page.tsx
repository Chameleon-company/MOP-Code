'use client';

import React, { useState } from 'react';
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { useTranslations } from "next-intl";
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import "../../../../public/styles/privacy.css";

interface AccordionProps {
  title: string;
  content: string;
  isOpen: boolean;
  onClick: () => void;
}

const AccordionItem: React.FC<AccordionProps> = ({ title, content, isOpen, onClick }) => (
  <div className="accordion bg-white border rounded-lg shadow-md transition-all duration-300 mb-4">
    <button
      onClick={onClick}
      className="w-full text-left p-4 font-bold text-green-600 flex justify-between items-center"
    >
      {title}
      <i className={`fas fa-chevron-${isOpen ? 'up' : 'down'}`} />
    </button>
    <div
      className="accordion-content bg-green-50 overflow-hidden transition-all duration-500 ease-in-out"
      style={{
        maxHeight: isOpen ? '200px' : '0',
        opacity: isOpen ? 1 : 0,
        padding: isOpen ? '1rem' : '0 1rem'
      }}
    >
      <p>{content}</p>
    </div>
  </div>
);

const sections = [
  { titleKey: 't1', contentKey: 'p1' },
  { titleKey: 't2', contentKey: 'p2' },
  { titleKey: 't3', contentKey: 'p3' },
  { titleKey: 't4', contentKey: 'p4' },
  { titleKey: 't5', contentKey: 'p5' },
  { titleKey: 't6', contentKey: 'p6' }
];

const Privacypolicy: React.FC = () => {
  const t = useTranslations("privacypolicy");
  const [openIndexes, setOpenIndexes] = useState<number[]>([]);
  const [searchTerm, setSearchTerm] = useState<string>("");

  const toggleAccordion = (index: number) => {
    setOpenIndexes(prev =>
      prev.includes(index) ? prev.filter(i => i !== index) : [...prev, index]
    );
  };

  const filteredSections = sections.filter((section) =>
    t(section.titleKey).toLowerCase().includes(searchTerm.toLowerCase())
  );

  const expandAll = () => setOpenIndexes(filteredSections.map((_, index) => index));
  const collapseAll = () => setOpenIndexes([]);

  const downloadPDF = () => {
    const input = document.querySelector('.policy-box');
    if (!input) return;

    html2canvas(input as HTMLElement).then((canvas: HTMLCanvasElement) => {
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const imgProps = pdf.getImageProperties(imgData);
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save('privacy-policy.pdf');
    });
  };

  return (
    <div className="bg-gray-300 min-h-screen">
      <Header />
      <main className="font-montserrat policy-box">
        <div className="ml-[5%] mt-[5%]">
          <h1 className="font-bold text-title">{t("Privacy Policy")}</h1>
        </div>

        <div className="flex flex-col items-center gap-4 mt-6 mx-[5%]">
          <input
            type="text"
            placeholder="Search sections..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full max-w-md p-2 border border-gray-300 rounded"
          />

          <div className="flex gap-4 flex-wrap justify-center">
            <button onClick={expandAll} className="bg-green-600 text-white px-4 py-2 rounded">Expand All</button>
            <button onClick={collapseAll} className="bg-green-600 text-white px-4 py-2 rounded">Collapse All</button>
            <button onClick={downloadPDF} className="bg-green-600 text-white px-4 py-2 rounded">Download PDF</button>
          </div>
        </div>

        <div className="mx-[5%] mt-6">
          {filteredSections.length === 0 ? (
            <p>No matching sections found.</p>
          ) : (
            filteredSections.map((section, index) => (
              <AccordionItem
                key={index}
                title={t(section.titleKey)}
                content={t(section.contentKey)}
                isOpen={openIndexes.includes(index)}
                onClick={() => toggleAccordion(index)}
              />
            ))
          )}
        </div>

        <div className="flex items-center justify-center mt-24 mb-[-3rem]">
          <p className="text-center text-[14px] w-[80%]">{t("p7")}</p>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Privacypolicy;
