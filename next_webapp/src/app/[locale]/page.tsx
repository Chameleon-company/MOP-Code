// Home.js
import Header from "../../components/Header";
import Footer from "../../components/Footer";
import Dashboard from "../../components/Dashboard";
import Chatbot from "../chatbot/chatbot";
import DashboardCaseStd from "@/components/DashboardCaseStd";

const Home = () => {
  return (
    <div>
      <Header />
      <Dashboard />
      <DashboardCaseStd />
      <Chatbot />
      <Footer />
    </div>
  );
};

export default Home;
