// Home.js
import Header from '../components/Header';
import Footer from '../components/Footer';
import Dashboard from '../components/Dashboard';
import TopicsList from '@/components/TopicsList';

const Home = () => {
  return (
    <div>
      <Header />
      <Dashboard /> 
      <TopicsList />
      <p>asdad</p>
      <Footer />
    </div>
  );
};

export default Home;