import type { Metadata } from "next";
import { getTranslations } from "next-intl/server";
import Header from "../../components/Header";
import Footer from "../../components/Footer";
import Dashboard from "../../components/Dashboard";
import Chatbot from "../chatbot/chatbot";
import ContactUsSection from "@/components/ContactUsSection";
import BackToTopButton from "@/components/BackToTopButton";
import PartnersSection from "@/components/PartnersSection";
import TestimonialsSection from "@/components/TestimonialsSection";
import Insights from "@/components/Insights";
import FAQSection from "@/components/FAQSection";
import SocialMediaFeed from "@/components/SocialMediaFeed";
import UseCaseInsights from "@/components/UseCaseInsights";
import Features from "@/components/Feature";
import { getHreflangAlternates } from "@/lib/seo/hreflang";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: "seo" });
  const { canonical, languages } = getHreflangAlternates("/");

  return {
    title: t("home_title"),
    description: t("home_description"),
    alternates: { canonical, languages },
  };
}

const Home = () => {
  return (
    <div>
      <Header />
      <Dashboard />
      <Insights />
      <UseCaseInsights />
      <Features />
      <TestimonialsSection />
      <FAQSection />
      <SocialMediaFeed />
      <ContactUsSection />
      <PartnersSection />
      <BackToTopButton />
      <Chatbot />
      <Footer />
    </div>
  );
};

export default Home;