"use client";
import React from "react";
import { useTranslations } from "next-intl";

export type Feature = {
  id: string | number;
  title: string;
};

type Props = {
  features?: Feature[];
};

const defaultFeatures: Feature[] = [
  { id: 1, title: "Chatbot Integration" },
  { id: 2, title: "Citizen Feedback & Reporting Tool" },
  { id: 3, title: "Personalised Recommendations & Alerts" },
  { id: 4, title: "Real-Time Data Insights Panel" },
];

const Features: React.FC<Props> = ({ features = defaultFeatures }) => {
  

  return (
    <section className="w-full bg-grey py-12 lg:py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        <h2 className="text-4xl sm:text-5xl font-extrabold text-center text-black mb-12">
          Features
        </h2>

        
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-8">
          {features.map((f) => (
            <div
              key={f.id}
              className="bg-gray-200 rounded-xl shadow-md flex items-center justify-center h-64 text-lg font-semibold text-black text-center p-4"
            >
              {f.title}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
