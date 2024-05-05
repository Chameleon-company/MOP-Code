// page.js
import Header from '../components/Header';
import Footer from '../components/Footer';
import Dashboard from '../components/Dashboard';
import ChatBot from '../app/chatbot/chatbot';

const Page: React.FC = () => {
  return (
    <div>
      <Header />
      <Dashboard />
      <ChatBot />
      <Footer />
    </div>
  );
};

export default Page;