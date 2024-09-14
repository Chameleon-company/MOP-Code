import React from "react";
import Link from "next/link";
import { useTranslations } from "next-intl";

const DashboardCaseStd = () => {
  const t = useTranslations("common");

  const caseStudies = [
    {
      id: "cs1",
      image: "/img/icon1.png",
      title: t("Case Study 1"),
      description: t("ChildCare Facilities Analysis"),
      link: "/UseCases",
    },
    {
      id: "cs2",
      image: "/img/icon2.png",
      title: t("Case Study 2"),
      description: t("Projected Venue Growth"),
      link: "/UseCases",
    },
    {
      id: "cs3",
      image: "/img/icon3.png",
      title: t("Case Study 3"),
      description: t("Data Science in Education"),
      link: "/UseCases",
    },
  ];

  return (
    <div className="case-studies-wrapper grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 p-4">
      {caseStudies.map((caseStudy) => (
        <Link href={`en/${caseStudy.link}`} key={caseStudy.id}>
          <div className="card-wrapper bg-white shadow-lg rounded-lg overflow-hidden hover:shadow-xl transition-shadow duration-300">
            <div className="top-image h-48 flex justify-center items-center bg-gray-100">
              <img src={caseStudy.image} alt={caseStudy.title} className="object-contain h-full" />
            </div>
            <div className="p-6">
              <h4 className="title text-xl font-semibold mb-3">{caseStudy.title}</h4>
              <p className="description text-gray-700">{caseStudy.description}</p>
            </div>
          </div>
        </Link>
      ))}
    </div>
  );
};

export default DashboardCaseStd;
