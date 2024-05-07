// Home.js
import Header from '../components/Header';
import Footer from '../components/Footer';
import Dashboard from '../components/Dashboard';
import DashboardCaseStd from '@/components/DashboardCaseStd';
import Chatbot from '../app/chatbot/chatbot';

const Home = () => {
  return (
    <>
      <Header />

      <Dashboard />
      <DashboardCaseStd /> 
      <Chatbot />
      <Footer />
    </>
  );
};

export default Home;