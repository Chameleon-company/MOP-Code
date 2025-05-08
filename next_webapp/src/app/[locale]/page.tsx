// Home.js
import Header from "../../components/Header";
import Footer from "../../components/Footer";
import Dashboard from "../../components/Dashboard";
import Chatbot from "../chatbot/chatbot";

const Home = () => {
  return (
    <div>
      <Header />
      <Dashboard /> 
      <Chatbot />
      <Footer />
    </div>
  );
};

export default Home;
