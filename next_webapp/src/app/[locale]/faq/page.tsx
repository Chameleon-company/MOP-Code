import React from 'react';
import Header from '../../../components/Header';
import Chatbot from '../../chatbot/chatbot';
import Footer from '../../../components/Footer';
// import styles from '../chatbot_faq/faqchatbot.module.css';


const FAQ = () => {
  return (
    <div className="flex flex-col min-h-screen bg-gray-100"> {/* This ensures that the footer stays at the bottom of the page */}
    
      <Header /> 

      <main className="flex-grow container mx-auto px-4 text-black">
        <header className="py-5">
          <h1 className="text-center text-3xl font-bold">FAQ - Chatbot Help</h1>
        </header>

        <section className="mb-10 w-full md:w-2/3">
          <h2 className="text-xl font-semibold mb-3">How to Use the Chatbot</h2>
          <p className="text-md mb-2">
            Here are some common questions and guides on how to interact with our chatbot effectively.
          </p>
          <ul className="list-disc pl-5">
            <li>Start by clicking the chat icon in the bottom right corner of your screen.</li>
            <li>Type your desired page in the input field that appears at the bottom of the chat window.</li>
            <li className='font-bold'>Eg: Take me to the usecase page, show me contact us page</li>
            <li>Press send or enter to submit your question.</li>
          </ul>
        </section>

        <section className="mb-10 w-full md:w-2/3">
          <h2 className="text-xl font-semibold mb-3">Common Issues</h2>
          <p className="text-md mb-2">
            If you are experiencing issues, here are some quick tips:
          </p>
          <ul className="list-disc pl-5">
            <li>Ensure your internet connection is stable.</li>
            <li>Refer to the specific commands listed in the chatbot description.</li>
            <li>Contact support if issues persist.</li>
          </ul>
        </section>

        
      </main>

      <Chatbot />
      
      <Footer /> 
    </div>
  );
};

export default FAQ;
