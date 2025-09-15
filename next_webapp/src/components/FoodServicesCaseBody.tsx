import React from "react";

type Props = {
  heroSrc?: string;
  charts?: { barSrc: string; lineSrc: string };
  wordCloudSrc?: string;
  title?: string;
  subtitle?: string;
  introTitle?: string;
  introParas?: string[];
  conclusion?: string;
};

export default function FoodServicesCaseBody({
  heroSrc = "public/img/P1.png",
  charts = { barSrc: "public/img/P2.png", lineSrc: "public/img/P3.png" },
  wordCloudSrc = "public/img/P3.png",
  title = "Food Services & Wellbeing",
  subtitle = "Understanding the impact of diet on lifestyle and wellness",
  introTitle = "Forecasting Hunger: A Decade Of Uncertainty",
  introParas = [
    "We analysed open-data on service listings and trends from 2013–2023. The data shows fluctuations with a notable spike in 2020–21.",
    "The bar chart summarises the most accessed service categories. The line chart shows the demand signal year-over-year. A keyword analysis is visualised with a word cloud.",
  ],
  conclusion = "Service discoverability, proximity, and eligibility clarity correlate with uptake. Improving signposting, extending hours, and keeping descriptions current can reduce unmet need.",
}: Props) {
  return (
    <article className="bg-white">
      {/* HERO */}
      <section className="relative h-64 md:h-80 lg:h-96 overflow-hidden rounded-b-2xl">
        <img
          src={heroSrc}
          alt="Food services hero"
          className="absolute inset-0 h-full w-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-black/20 to-transparent" />
        <div className="relative z-10 h-full max-w-6xl mx-auto px-6 md:px-10 flex flex-col justify-end pb-6">
          <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold text-emerald-300 drop-shadow">
            {title}
          </h1>
          <p className="text-white/90 text-base md:text-lg mt-2 max-w-2xl drop-shadow">
            {subtitle}
          </p>
        </div>
      </section>

      {/* CONTENT */}
      <section className="relative bg-gray-50">
        <div className="max-w-5xl mx-auto px-6 md:px-10 py-10">
          <h2 className="text-xl md:text-2xl font-semibold text-gray-900 mb-4">
            {introTitle}
          </h2>
          {introParas.map((p, i) => (
            <p key={i} className="text-gray-700 leading-7 mb-4">
              {p}
            </p>
          ))}

          {/* CHARTS */}
          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-8">
            <figure>
              <img
                src={charts.barSrc}
                alt="Accessed services by category"
                className="w-full rounded-xl border border-gray-200 shadow-sm bg-white"
              />
              <figcaption className="mt-2 text-sm text-gray-500">
                Accessed services by category
              </figcaption>
            </figure>
            <figure>
              <img
                src={charts.lineSrc}
                alt="Annual demand signal"
                className="w-full rounded-xl border border-gray-200 shadow-sm bg-white"
              />
              <figcaption className="mt-2 text-sm text-gray-500">
                Annual demand signal
              </figcaption>
            </figure>
          </div>

          {/* WORD CLOUD */}
          <div className="mt-10">
            <figure>
              <img
                src={wordCloudSrc}
                alt="Most quoted words"
                className="w-full rounded-xl border border-gray-200 shadow-sm bg-white"
              />
              <figcaption className="mt-2 text-sm text-gray-500">
                Most quoted words in service descriptions
              </figcaption>
            </figure>
          </div>

          {/* CONCLUSION */}
          <div className="mt-10">
            <h3 className="text-lg md:text-xl font-semibold text-gray-900 mb-3">
              Conclusion
            </h3>
            <p className="text-gray-700 leading-7">{conclusion}</p>
          </div>
        </div>
      </section>
    </article>
  );
}
