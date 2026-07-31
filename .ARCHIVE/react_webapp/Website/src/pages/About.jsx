import React from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import '../styles/about.css';

const About = () => {
  return (
    <div>
      <Header />
      <main className="main-content">
        <div className="featured-text">
          <h1>About us</h1>
          <p>
            Melbourne Open Data Project (MOP) is a capstone project sponsored by
            Deakin University. Since COVID, there has been an increased demand for
            data by the business community to support their decision-making. This
            project is meant to align with two strategic documents from the
            Melbourne City Council.
          </p>
        </div>
      </main>
      <div className="info-section">
        <div className="info-block">
          <div className="additional-logo-container">
            <img src="src/assets/about-logo 2.png" alt="Additional About Us Logo" />
          </div>
          <h2>About us</h2>
          <p>
            This project is meant to align with two strategic documents from the
            Melbourne City Council: The Economic Development Strategy, which aims to
            be a digitally-connected city. The 2021-2025 Council Plan, which
            outlines the specific objective of delivering programs that will build
            digital literacy skills and capabilities.
          </p>
        </div>
        <div className="info-block">
          <h2>Open Data Leadership</h2>
          <p>
            The City of Melbourne has been an Australian leader in Open Data since
            2014. Recent research and local user engagement have identified a gap
            where users would like to learn more about how to access Open Data and
            how to gain insights from the data to build apps and solve city
            problems.
          </p>
        </div>
        <div className="info-block">
          <h2>Our Goals</h2>
          <p>
            This project aims to deliver proof-of-concept examples on how calls to
            Open Data APIs can be made.
          </p>
          <div className="main-logo-container">
            <img src="src/assets/about-logo.png" alt="Melbourne Open Data Project Logo" />
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default About;
