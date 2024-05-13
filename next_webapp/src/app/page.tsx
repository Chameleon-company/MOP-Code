// Home.js
import Header from '../components/Header';
import Footer from '../components/Footer';
import Dashboard from '../components/Dashboard';
import Chatbot from '../app/chatbot/chatbot';


const Home = () => {
  return (
    <div className='w-full'>
      <Header />
      <Dashboard /> 
      <Chatbot />
      <Footer />
    </div>
  );
};

export default Home;
