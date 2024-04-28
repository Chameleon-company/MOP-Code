// Home.js
import Header from '../components/Header';
import Footer from '../components/Footer';
import Dashboard from '../components/Dashboard';
import DashboardCaseStd from '@/components/DashboardCaseStd';

const Home = () => {
  return (
    <>
      <Header />
      <Dashboard />
      <DashboardCaseStd />
      <Footer />
    </>
  );
};

export default Home;