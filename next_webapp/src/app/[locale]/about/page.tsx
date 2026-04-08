
"use client";

import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/about.css";
import { useTranslations } from "next-intl";
import Link from "next/link";

const About = () => {
  const t = useTranslations("about");

  return (
    <div className="bg-white dark:bg-[#1d1919] text-black dark:text-white min-h-screen">

      <Header />

      {/* HERO SECTION */}
      <section className="section max-w-6xl mx-auto flex flex-col md:flex-row items-center gap-10">
        
        {/* Image */}
       <img
        src="/img/melbourne-city1.jpg"
        alt="Melbourne City"
        className="hero-img w-3/4 md:w-2/5 h-auto"
        />

        {/* Text */}
        <div className="md:w-1/2">
          <h1 className="section-title">About Us</h1>
          <p className="section-subtitle">
            The Melbourne Open Data Project (MOP) is a capstone initiative aligned
            with the City of Melbourne’s strategic vision. It transforms open data
            into actionable insights using AI, data science, and modern web
            technologies.
          </p>
        </div>
      </section>

      {/* PROJECT OVERVIEW */}
      <section className="section bg-gray-100 dark:bg-[#263238] text-center">
        <h2 className="section-title">Project Overview</h2>
        <p className="section-subtitle max-w-3xl mx-auto">
          MOP enables businesses, researchers, and government agencies to explore
          real time urban data, AI driven insights, and visualisations. The platform
          supports smarter decision making across sustainability, transport,
          healthcare, and economic development.
        </p>
      </section>

      {/* OBJECTIVES */}
      <section className="section max-w-6xl mx-auto">
        <h2 className="section-title text-center">Our Objectives</h2>

        <img
          src="/img/objectives.jpg"
          alt="Objectives"
          className="hero-img w-full h-[280px] md:h-[320px] object-cover"
        />

        <div className="grid md:grid-cols-3 gap-8 mt-8">
    
          <div className="card bg-white/70 backdrop-blur-md border border-gray-200 shadow-md rounded-xl p-6 hover:shadow-xl transition">
            <h3 className="font-semibold text-xl mb-2">Data Accessibility</h3>
            <p>Make open data easy to access and understandable for all users.</p>
          </div>

          <div className="card bg-white/70 backdrop-blur-md border border-gray-200 shadow-md rounded-xl p-6 hover:shadow-xl transition">
            <h3 className="font-semibold text-xl mb-2">Smart Insights</h3>
            <p>Use AI and analytics to generate meaningful insights from complex datasets.</p>
          </div>

          <div className="card bg-white/70 backdrop-blur-md border border-gray-200 shadow-md rounded-xl p-6 hover:shadow-xl transition">
            <h3 className="font-semibold text-xl mb-2">Urban Innovation</h3>
            <p>Support smart city initiatives and improve urban living experiences.</p>
          </div>

      </div>
      </section>

      {/* KEY FEATURES */}
      <section className="section bg-gradient-to-r from-green-500 to-emerald-600 text-white py-16 px-6">

        <h2 className="section-title text-center text-3xl font-bold">
          Key Features
        </h2>

        <div className="grid md:grid-cols-4 gap-6 mt-10 max-w-6xl mx-auto">

          <div className="feature-card bg-white/15 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/25 hover:scale-105 transition duration-300 shadow-lg">
            <h4 className="font-bold text-lg mb-2">Real-time Data</h4>
            <p className="text-sm text-white/90">Live updates from urban datasets.</p>
          </div>

          <div className="feature-card bg-white/15 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/25 hover:scale-105 transition duration-300 shadow-lg">
            <h4 className="font-bold text-lg mb-2">AI Analytics</h4>
            <p className="text-sm text-white/90">Predictive and intelligent insights.</p>
          </div>

          <div className="feature-card bg-white/15 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/25 hover:scale-105 transition duration-300 shadow-lg">
            <h4 className="font-bold text-lg mb-2">Interactive UI</h4>
            <p className="text-sm text-white/90">User-friendly dashboards and visualisations.</p>
          </div>

          <div className="feature-card bg-white/15 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/25 hover:scale-105 transition duration-300 shadow-lg">
            <h4 className="font-bold text-lg mb-2">Open APIs</h4>
            <p className="text-sm text-white/90">Seamless integration with public data sources.</p>
          </div>

        </div>
      </section>

     {/* CTA SECTION */}
      <section className="section text-center">
        <h2 className="section-title">Explore Our Platform</h2>

        <p className="section-subtitle">
          Discover how data driven solutions can transform industries and improve city life.
        </p>

        <Link href="/usecases">
          <button className="cta-btn mt-6">
            View Use Cases
          </button>
        </Link>
      </section>

      <Footer />

    </div>
  );
};

export default About;