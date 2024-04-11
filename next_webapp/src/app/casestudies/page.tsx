"use client"
import React, { useState, useEffect } from 'react';
import Header from "../../components/Header";
import Footer from "../../components/Footer";
import SearchBar from '../casestudies/searchbar'; 
import PreviewComponent from '../casestudies/preview';

// Adjust the path based on your project structure
const CaseStudies = () => {
  
 const [images, setImages] = useState([]);
 const [selectedStudy, setSelectedStudy] = useState(null);
 const caseStudies = [
   { id: 1, image: '/img/one.png', title: 'Case Study 1', details: 'Through the lens of Melbourne open data, data science in biotech unveils a fascinating narrative, bridging innovative advancements in biotechnology with rich, publicly accessible datasets, illuminating novel possibilities and solutions within the realm of health sciences and biotech innovation' },
   { id: 2, image: '/img/two.png', title: 'Case Study 2', details: 'Within the domain of predictive modeling for sustaining oil and gas supply chains, leveraging Melbournes open data enriches the analysis, offering a diverse perspective that enhances forecasting accuracy, operational efficiency, and informed decision-making within this critical sector.' },
   { id: 3, image: '/img/three.png', title: 'Case Study 3', details: 'Delvings into the realm of education through data science, the utilization of Melbourne open data becomes a catalyst for transformative insights and informed strategies, fostering innovation and enhancing educational practices within the city dynamic learning landscape.' },
 ];
 useEffect(() => {
   setImages(caseStudies);
 }, []);
 const openModal = (study) => {
   setSelectedStudy(study);
 };
 const closeModal = () => {
   setSelectedStudy(null);
 };
 const handleSearch = (query) => {
  // Here you might want to route to a new page with the search query
  // For example, using Next.js router.push
  //router.push(`/search?q=${query}`);
};
 return (
<div className="font-sans bg-gray-100">
<Header />
<main>
<div className="app">
<section >
<p><span className="text-4xl text-black">Case</span></p>
<SearchBar onSearch={handleSearch} />
<PreviewComponent />
</section>
</div>
</main>
<Footer />
</div>
 );
};

export default CaseStudies; 