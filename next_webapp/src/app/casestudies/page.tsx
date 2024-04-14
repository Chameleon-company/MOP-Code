// App.tsx
"use client"; // This is a client component 
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import '../casestudies/index.css'; // Ensuring this points to the correct path where Tailwind CSS is imported
import '@fortawesome/fontawesome-free/css/all.min.css';
import Header from '../casestudies/Header'; // Adjusting the import path according to the directory structure
import CaseStudies from '../casestudies/CaseStudies'; // This component includes the search functionality and the case studies list
import Footer from '../casestudies/Footer'; 

const App: React.FC = () => {
  return (
    <Router>
      <div className="App">
        <Header />
        <main className="mt-14">
          <Routes>
            <Route path="/" element={<CaseStudies />} />
            {/* other routes are defined here */}
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
};

export default App;
