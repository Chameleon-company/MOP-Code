import Header from "../../components/Header";
import Footer from "../../components/Footer";
import Dashboard from "../../components/Dashboard";
import Chatbot from "../chatbot/chatbot";
import SocialMediaFeed from "@/components/SocialMediaFeed";
import CityMetricCard from "@/components/CityMetricCard";
import Insights from "@/components/Insights";

const Home = () => {
  return (
    <div>
      <Header />
      <Dashboard />
      <Insights />

      <SocialMediaFeed />

      <Chatbot />
      <Footer />
    </div>
  );
};

export default Home;
