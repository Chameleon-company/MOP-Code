// Home.js
import React from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
// import HeaderBackup from '../components/HeaderBackup';
import Dashboard from '../components/Dashboard';

const Home = () => {
  return (
    <div>
      <Header />
      <Dashboard />
      <Footer />
    </div>
  );
};

export default Home;