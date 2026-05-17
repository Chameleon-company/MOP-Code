"use client";

import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/about.css";
import { useTranslations } from "next-intl";
import { Link } from "@/i18n-navigation";

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
						The Melbourne Open Data Project (MOP) is a capstone initiative
						aligned with the City of Melbourne’s strategic vision. It transforms
						open data into actionable insights using AI, data science, and
						modern web technologies.
					</p>
				</div>
			</section>

			{/* PROJECT OVERVIEW */}
			<section className="section bg-gray-100 dark:bg-[#263238] text-center">
				<h2 className="section-title">Project Overview</h2>
				<p className="section-subtitle max-w-3xl mx-auto">
					MOP enables businesses, researchers, and government agencies to
					explore real time urban data, AI driven insights, and visualisations.
					The platform supports smarter decision making across sustainability,
					transport, healthcare, and economic development.
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
						<p>
							Make open data easy to access and understandable for all users.
						</p>
					</div>

					<div className="card bg-white/70 backdrop-blur-md border border-gray-200 shadow-md rounded-xl p-6 hover:shadow-xl transition">
						<h3 className="font-semibold text-xl mb-2">Smart Insights</h3>
						<p>
							Use AI and analytics to generate meaningful insights from complex
							datasets.
						</p>
					</div>

					<div className="card bg-white/70 backdrop-blur-md border border-gray-200 shadow-md rounded-xl p-6 hover:shadow-xl transition">
						<h3 className="font-semibold text-xl mb-2">Urban Innovation</h3>
						<p>
							Support smart city initiatives and improve urban living
							experiences.
						</p>
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
						<p className="text-sm text-white/90">
							Live updates from urban datasets.
						</p>
					</div>

					<div className="feature-card bg-white/15 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/25 hover:scale-105 transition duration-300 shadow-lg">
						<h4 className="font-bold text-lg mb-2">AI Analytics</h4>
						<p className="text-sm text-white/90">
							Predictive and intelligent insights.
						</p>
					</div>

					<div className="feature-card bg-white/15 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/25 hover:scale-105 transition duration-300 shadow-lg">
						<h4 className="font-bold text-lg mb-2">Interactive UI</h4>
						<p className="text-sm text-white/90">
							User-friendly dashboards and visualisations.
						</p>
					</div>

					<div className="feature-card bg-white/15 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/25 hover:scale-105 transition duration-300 shadow-lg">
						<h4 className="font-bold text-lg mb-2">Open APIs</h4>
						<p className="text-sm text-white/90">
							Seamless integration with public data sources.
						</p>
					</div>
				</div>
			</section>

			{/* CTA SECTION */}
			<section className="section text-center">
				<h2 className="section-title">Explore Our Platform</h2>

				<p className="section-subtitle">
					Discover how data driven solutions can transform industries and
					improve city life.
				</p>

				<Link href="/usecases" className="inline-block mt-6">
					<button type="button" className="cta-btn">
						View Use Cases
					</button>
				</Link>
			</section>

			<Footer />
		</div>
	);
  const t = useTranslations("about");
  const [darkMode, setDarkMode] = useState(false);

  // useEffect(() => {
  //   const root = document.documentElement;
  //   darkMode ? root.classList.add("dark") : root.classList.remove("dark");
  // }, [darkMode]);

  useEffect(() => {
  const saved = localStorage.getItem("darkMode");
  if (saved) setDarkMode(JSON.parse(saved));
}, []);

  useEffect(() => {
    const root = document.documentElement;
    darkMode ? root.classList.add("dark") : root.classList.remove("dark");
    localStorage.setItem("darkMode", JSON.stringify(darkMode));
  }, [darkMode]);


  const Section = ({
    imageSrc,
    imageAlt,
    title,
    text,
    bgClass,
  }: {
    imageSrc: string;
    imageAlt: string;
    title: string;
    text: string;
    bgClass: string;
  }) => (
    <section className={`${bgClass} py-12`}>
      <div className="max-w-6xl mx-auto px-4 flex flex-col md:flex-row items-center md:items-center gap-8 min-h-[400px]">
        {/* Image */}
        <div className="w-full md:w-1/2">
          <img
            src={imageSrc}
            alt={imageAlt}
            className="w-full aspect-video object-cover rounded-lg shadow-md"
          />
        </div>

        {/* Text Content */}
        <div className="w-full md:w-1/2 flex flex-col justify-center">
          <h2 className="text-2xl font-bold mb-4 text-black dark:text-white">{title}</h2>
          <p className="text-base text-justify text-black dark:text-white leading-relaxed">
            {text}
          </p>
        </div>
      </div>
    </section>
  );

  return (
    <div className="bg-white dark:bg-[#1d1919] text-black dark:text-white min-h-screen transition-colors duration-300">
      <Header />

      {/* Section 1: About Us – light: white | dark: #263238 */}
      <Section
        imageSrc="/img/mel.jpg"
        imageAlt="Melbourne Open Playground"
        title={t("About Us")}
        text={t("p2")}
        bgClass="bg-white dark:bg-[#263238]"
      />

      {/* Section 2: Open Data Leadership – light: green-500 | dark: #14532d */}
      <Section
        imageSrc="/img/leadership.png"
        imageAlt="Leadership Image"
        title={t("Open Data Leadership")}
        text={t("p3")}
        bgClass="bg-green-500 dark:bg-[#14532d]"
      />

      {/* Section 3: Our Goals – light: white | dark: #263238 */}
      <Section
        imageSrc="/img/goals.png"
        imageAlt="Our Goals"
        title={t("Our Goals")}
        text={t("p4")}
        bgClass="bg-white dark:bg-[#263238]"
      />

      <Footer />
    </div>
  );
};

export default About;
