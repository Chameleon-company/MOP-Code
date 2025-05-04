// Home.js
import Header from "../../components/Header";
import Footer from "../../components/Footer";
import Dashboard from "../../components/Dashboard";
import Chatbot from "../chatbot/chatbot";
import DashboardCaseStd from "@/components/DashboardCaseStd";
import SocialMediaFeed from "@/components/SocialMediaFeed";

const Home = () => {
  return (
    <div>
      <Header />
      <Dashboard />
      <DashboardCaseStd />
      <SocialMediaFeed />
      <Chatbot />
      <Footer />
    </div>
  );
};

export default Home;
