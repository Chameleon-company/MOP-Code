// Home.js
import Header from "../../components/Header";
import Footer from "../../components/Footer";
import Dashboard from "../../components/Dashboard";
import Chatbot from "../chatbot/chatbot";
import SocialMediaFeed from "@/components/SocialMediaFeed";
import UseCaseInsights from "@/components/UseCaseInsights";


const Home = () => {
  return (
    <div>
      <Header />
      <Dashboard />
      <UseCaseInsights />
      <SocialMediaFeed />
      <Chatbot />
      <Footer />
    </div>
  );
};

export default Home;
