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