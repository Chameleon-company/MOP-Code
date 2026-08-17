import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import ContributorsSection from "../../../components/ContributorsSection";
import "./about.css";
import { Link } from "@/i18n-navigation";
import { getContributors } from "@/lib/getContributors";
import Image from "next/image";
import { getTranslations } from "next-intl/server";

const About = async () => {
  const contributors = (await getContributors()) ?? [];
  const t = await getTranslations("about");

  return (
    <div className="min-h-screen bg-white text-black dark:bg-[#1d1919] dark:text-white">
      <Header />

      {/* HERO SECTION */}
      <section className="section mx-auto flex max-w-6xl flex-col items-center gap-10 md:flex-row">
        <Image
          src="/img/melbourne-city1.jpg"
          alt="Melbourne City"
          width={1080}
          height={1350}
          sizes="(max-width: 768px) 75vw, 40vw"
          className="hero-img h-auto w-full sm:w-3/4 md:w-2/5 lg:w-1/3"
        />

        <div className="md:w-1/2">
          <h1 className="section-title">
            {t("About Us")}
          </h1>

          <p className="section-subtitle">
            {t("heroDescription")}
          </p>
        </div>
      </section>

      {/* PROJECT OVERVIEW */}
      <section className="section bg-gray-100 text-center dark:bg-[#263238]">
        <h2 className="section-title">
          {t("Project Overview")}
        </h2>

        <p className="section-subtitle mx-auto max-w-3xl">
          {t("projectOverviewDescription")}
        </p>
      </section>

      {/* OBJECTIVES */}
      <section className="section mx-auto max-w-6xl">
        <h2 className="section-title text-center">
          {t("Our Objectives")}
        </h2>

        <Image
          src="/img/objectives.jpg"
          alt="Objectives"
          width={5402}
          height={3601}
          sizes="(max-width: 768px) 100vw, 1152px"
          className="hero-img h-[220px] w-full object-cover sm:h-[280px] md:h-[320px] lg:h-[360px]"
        />

        <div className="mt-8 grid gap-8 md:grid-cols-3">
          <div className="card rounded-xl border border-gray-200 bg-white/70 p-6 shadow-md backdrop-blur-md transition hover:shadow-xl">
            <h3 className="mb-2 text-xl font-semibold">
              {t("Data Accessibility")}
            </h3>

            <p>
              {t("dataAccessibilityDescription")}
            </p>
          </div>

          <div className="card rounded-xl border border-gray-200 bg-white/70 p-6 shadow-md backdrop-blur-md transition hover:shadow-xl">
            <h3 className="mb-2 text-xl font-semibold">
              {t("Smart Insights")}
            </h3>

            <p>
              {t("smartInsightsDescription")}
            </p>
          </div>

          <div className="card rounded-xl border border-gray-200 bg-white/70 p-6 shadow-md backdrop-blur-md transition hover:shadow-xl">
            <h3 className="mb-2 text-xl font-semibold">
              {t("Urban Innovation")}
            </h3>

            <p>
              {t("urbanInnovationDescription")}
            </p>
          </div>
        </div>
      </section>

      {/* KEY FEATURES */}
      <section className="section bg-gradient-to-r from-green-500 to-emerald-600 px-6 py-16 text-white">
        <h2 className="section-title text-center text-3xl font-bold">
          {t("Key Features")}
        </h2>

        <div className="mx-auto mt-10 grid max-w-6xl gap-6 md:grid-cols-4">
          <div className="feature-card rounded-xl border border-white/20 bg-white/15 p-6 shadow-lg backdrop-blur-md transition duration-300 hover:scale-105 hover:bg-white/25">
            <h4 className="mb-2 text-lg font-bold">
              {t("Real-time Data")}
            </h4>

            <p className="text-sm text-white/90">
              {t("realTimeDataDescription")}
            </p>
          </div>

          <div className="feature-card rounded-xl border border-white/20 bg-white/15 p-6 shadow-lg backdrop-blur-md transition duration-300 hover:scale-105 hover:bg-white/25">
            <h4 className="mb-2 text-lg font-bold">
              {t("AI Analytics")}
            </h4>

            <p className="text-sm text-white/90">
              {t("aiAnalyticsDescription")}
            </p>
          </div>

          <div className="feature-card rounded-xl border border-white/20 bg-white/15 p-6 shadow-lg backdrop-blur-md transition duration-300 hover:scale-105 hover:bg-white/25">
            <h4 className="mb-2 text-lg font-bold">
              {t("Interactive UI")}
            </h4>

            <p className="text-sm text-white/90">
              {t("interactiveUIDescription")}
            </p>
          </div>

          <div className="feature-card rounded-xl border border-white/20 bg-white/15 p-6 shadow-lg backdrop-blur-md transition duration-300 hover:scale-105 hover:bg-white/25">
            <h4 className="mb-2 text-lg font-bold">
              {t("Open APIs")}
            </h4>

            <p className="text-sm text-white/90">
              {t("openAPIsDescription")}
            </p>
          </div>
        </div>
      </section>

      {/* CONTRIBUTORS */}
      <ContributorsSection contributors={contributors} />

      {/* CTA SECTION */}
      <section className="section text-center">
        <h2 className="section-title">
          {t("Explore Our Platform")}
        </h2>

        <p className="section-subtitle">
          {t("exploreDescription")}
        </p>

        <Link href="/usecases" className="mt-6 inline-block">
          <button type="button" className="cta-btn">
            {t("View Use Cases")}
          </button>
        </Link>
      </section>

      <Footer />
    </div>
  );
};

export default About;