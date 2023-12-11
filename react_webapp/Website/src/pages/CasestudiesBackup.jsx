import React, {useState} from 'react';
import '../styles/CasestudiesBackup.css';
import Header from '../components/Header';
import Footer from '../components/Footer';

const generateRandomData = () => {
  return Array.from({ length: 12 }, () => Math.floor(Math.random() * 100) + 20);
};

const CaseStudy = ({ title, description }) => {
  const [minimized, setMinimized] = useState(true);

  const toggleMinimized = () => {
    setMinimized(!minimized);
  };

  const chartData = generateRandomData();
  return (
    <div className="case-study" onClick={() => setMinimized(!minimized)}>
      <div className="case-study-header">
        <h2>{title}</h2>
        <button
          className="minimize-button"
          onClick={(e) => {
            e.stopPropagation();
            toggleMinimized();
          }}
        >
          -
        </button>
      </div>
      {!minimized && (
        <div className="case-study-content">
          <p>{description}</p>
          <svg width="300" height="150">
            {chartData.map((value, index) => (
              <rect
                key={index}
                x={index * 40}
                y={150 - value}
                width="25"
                height={value}
                fill="steelblue"
              />
            ))}
          </svg>
        </div>
      )}
    </div>
  );
};

const CasestudiesBackup = () => {
  const initialcasestudies = [
    { title: 'Case Study 1: Property', description: 'Random description for Case Study 1. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.' },
    { title: 'Case Study 2: People', description: 'Another random description for Case Study 2. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.' },
    { title: 'Case Study 3: Environment', description: 'Random description for Case Study 3. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.' },
    { title: 'Case Study 4: Transportation', description: 'Random description for Case Study 4. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.' },
    { title: 'Case Study 5: Sensors', description: 'Random description for Case Study 5. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.' },

  ];
  const [caseStudies, setCaseStudies] = useState(initialcasestudies);
  const [searchTerm, setSearchTerm] = useState('');

  const handleSearch = (e) => {
    const searchTerm = e.target.value.toLowerCase();
    setSearchTerm(searchTerm);

    const filteredCaseStudies = initialcasestudies.filter((study) =>
      study.title.toLowerCase().includes(searchTerm)
    );

    setCaseStudies(filteredCaseStudies);
  };
  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth',
    });
  };
  return (
    <div>
      <Header />
        <div className='parent'>
          <h2 className='page-title'>Case Studies</h2>
          <div className='search-bar'>
            <input
            type="text"
            placeholder="Search by title..."
            value={searchTerm}
            onChange={handleSearch}
            />
          </div>
        </div>
        <div>
          {caseStudies.map((study, index) => (
          <CaseStudy key={index} title={study.title} description={study.description} />
          ))}
        </div>
        <button className="back-to-top-button" onClick={scrollToTop}>
        ⬆
        </button>
      <Footer />
    </div>
  );
};

export default CasestudiesBackup;