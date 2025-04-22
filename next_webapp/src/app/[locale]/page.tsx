// Home.js
import Header from "../../components/Header";
import Footer from "../../components/Footer";
import Dashboard from "../../components/Dashboard";
import Chatbot from "../chatbot/chatbot";
import DashboardCaseStd from "@/components/DashboardCaseStd";
// import ThemeToggle from "@/components/ThemeToggle";

const Home = () => {
  return (
<div className="bg-white dark:bg-[#263238] text-black dark:text-white">
  <Header />
  <Dashboard />
  <DashboardCaseStd />
  <Chatbot />
  <Footer />
</div>
  );
};

export default Home;
