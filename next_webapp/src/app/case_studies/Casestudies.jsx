import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import imageOne from '../images/one.png';
import imageTwo from '../images/two.png';
import imageThree from '../images/three.png';
// Import Tailwind CSS styles
import 'tailwindcss/tailwind.css'; 
// Adjust the path based on your project structure
const CaseStudies = () => {
 const [images, setImages] = useState([]);
 const [selectedStudy, setSelectedStudy] = useState(null);
 const caseStudies = [
   { id: 1, image: imageOne, title: 'Case Study 1', details: 'Through the lens of Melbourne open data, data science in biotech unveils a fascinating narrative, bridging innovative advancements in biotechnology with rich, publicly accessible datasets, illuminating novel possibilities and solutions within the realm of health sciences and biotech innovation' },
   { id: 2, image: imageTwo, title: 'Case Study 2', details: 'Within the domain of predictive modeling for sustaining oil and gas supply chains, leveraging Melbournes open data enriches the analysis, offering a diverse perspective that enhances forecasting accuracy, operational efficiency, and informed decision-making within this critical sector.' },
   { id: 3, image: imageThree, title: 'Case Study 3', details: 'Delving into the realm of education through data science, the utilization of Melbourne open data becomes a catalyst for transformative insights and informed strategies, fostering innovation and enhancing educational practices within the city dynamic learning landscape.' },
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
 return (
<div className="font-sans bg-gray-100">
<Header />
<main>
<div className="app">
<section className="intro max-w-3/4 mx-auto bg-green-800 text-white p-4 rounded-lg">
<p><span className="text-4xl">Chameleon Melbourne Open Data</span> - <span className="text-xl font-bold">We unveil the intricate tapestry of data science's transformative impact across diverse sectors. Through compelling case studies exploring biotechnology, oil and gas supply management, and education, we showcase the power of Melbourne's open data. These narratives illuminate innovative solutions, empowering informed decision-making, and driving progress within our city's dynamic landscape.</span></p>
</section>
<section className="recent-work">
<h2 className="text-2xl">Recent Work</h2>
<div className="grid grid-cols-3 gap-4">
             {images.map((study) => (
<div key={study.id} className="case-study">
<button onClick={() => openModal(study)}>
<img src={study.image} alt={study.title} className="w-full h-auto" />
<p className="text-lg font-bold">{study.title}</p>
</button>
</div>
             ))}
</div>
</section>
         {selectedStudy && (
<div className="modal bg-gray-100 p-4 rounded-lg max-w-1/2 mx-auto">
<div className="modal-content">
<span className="close" onClick={closeModal}>
&times;
</span>
<h2>{selectedStudy.title}</h2>
<p>{selectedStudy.details}</p>
               {/* You can add more details here */}
</div>
</div>
         )}
<section className="outro max-w-3/4 mx-auto bg-green-800 text-white p-4 rounded-lg">
<p className="text-2xl font-bold">"Chameleon Melbourne Open Data stands as a testament to the profound possibilities unlocked by leveraging Melbourne's open data resources. From revolutionizing biotech insights to optimizing oil and gas supply chains and redefining educational strategies, these case studies highlight the integral role of data science in shaping a progressive future. Join us in exploring the boundless opportunities and transformative potential embedded within Melbourne's rich repository of open data."</p>
</section>
</div>
</main>
<Footer />
</div>
 );
};

export default CaseStudies;