// page.tsx
import React from 'react';
import '../../../public/styles/usecase.css'; // Ensuring this points to the correct path where Tailwind CSS is imported
import '@fortawesome/fontawesome-free/css/all.min.css'; // For using FontAwesome icons
import Header from '../../components/Header'; // Adjusting the import path according to the directory structure
import CaseStudies from '../UseCases/usecase'; // This component includes the search functionality and the case studies list
import Footer from '../../components/Footer'; 


const App: React.FC = () => {
  return (
    <div className="App">
      <Header />
      <main className="bg-white">
        <CaseStudies />
      </main>
      <Footer />
    </div>
  );
};

export default App;