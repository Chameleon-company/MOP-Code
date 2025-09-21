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
  heroSrc = "/img/P0.jpg", 
  charts = { barSrc: "/img/P2.png", lineSrc: "/img/P3.png" },
  wordCloudSrc = "/img/P3.png", 
  title = "Food Services & Wellbeing",
  subtitle = "Understanding the impact of diet on lifestyle and wellness",
  introTitle = "Forecasting Hunger: A Decade Of Uncertainty",
  introParas = [
    "This line graph offers a sobering look at food insecurity trends from 2013 to 2027. The historical data shows fluctuations, with a notable dip in 2015 and a peak in 2014. The forecasted values from 2018 onward suggest continued instability, raising concerns about future access to food.",
    "However, this model comes with limitations. It does not account for macroeconomic disruptions like inflation, the COVID-19 pandemic, or shifts in global supply chains—all of which have deeply impacted household food access. These omissions mean the forecast may underestimate the severity of future challenges.",
    "Still, the message is clear: cities must act proactively. A resilient food support system is essential. Every resident deserves reliable access to nutritious food, and forecasting tools like this can help policymakers anticipate needs and allocate resources effectively. Because no one should ever feel that their basic needs are being overlooked."
  ],
  conclusion = "With only 17 food services currently available in Melbourne and food insecurity projected to persist around the 6% mark, it is crucial for Melbourne to consider expanding and enhancing food services throughout the city. Ensuring adequate food access is not just a matter of convenience; it is essential for personal safety and security. Reliable access to food supports well-being, reduces stress, and fosters a safer, more resilient community. Addressing these needs proactively can make a significant difference in improving the quality of life for all residents.",
}: Props): React.JSX.Element {
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
