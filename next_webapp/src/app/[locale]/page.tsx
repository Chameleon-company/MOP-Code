import Header from "../../components/Header";
import Footer from "../../components/Footer";
import Dashboard from "../../components/Dashboard";
import Chatbot from "../chatbot/chatbot";
import SocialMediaFeed from "@/components/SocialMediaFeed";
import PartnersSection from "@/components/PartnersSection";
import TestimonialsSection from "@/components/TestimonialsSection";
import CityMetricCard from "@/components/CityMetricCard";
import Insights from "@/components/Insights";
import BackToTopButton from "@/components/BackToTopButton";

const Home = () => {
  return (
    <div>
      <Header />
      <Dashboard />
      <Insights />
      
      <TestimonialsSection />
      <PartnersSection />
      <SocialMediaFeed />
      <Chatbot />
      <BackToTopButton />
      <Footer />

    </div>
  );
};

export default Home;
