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
    <div className="case-studies-wrapper p-4">
      {caseStudies.map((caseStudy) => (
        <Link href={`en${caseStudy.link}`} key={caseStudy.id}>
          <a>
            <div className="card-wrapper bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md hover:shadow-xl transition-shadow duration-300">
              <div className="top-image mb-4">
                <img
                  src={caseStudy.image}
                  alt={caseStudy.title}
                  className="w-full h-40 object-cover rounded-lg"
                />
              </div>
              <h4 className="title text-xl font-semibold text-black dark:text-white">
                {caseStudy.title}
              </h4>
              <p className="description text-gray-700 dark:text-gray-300">
                {caseStudy.description}
              </p>
            </div>
          </a>
        </Link>
      ))}
    </div>
  );
};

export default DashboardCaseStd;
