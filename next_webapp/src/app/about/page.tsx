import Header from "../../components/Header";
import Footer from "../../components/Footer";
import '../../../public/styles/about.css';

const About = () => {
  return (
    <div className="bg-white">
            <Header />
            
    <div className="about-heading">About</div>
    <div className="about-heading">Us</div>
    <div className="image-container">
      <img src="/img/mel.jpg" alt="About Us Image"/>
    </div>

    <div className="banner">
    <h2>About MOP</h2>
    <p>Melbourne Open Data Project (MOP) is a capstone project sponsored by Deakin University in collaboration with City of Melbourne. Since COVID, there has been an increased demand for data by the business community to support their decision-making. This project is meant to align with two strategic documents from the Melbourne City Council.</p>
  </div>
  
            <div className="m-4 bg-white">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mx-10 mt-10">
                    <div style={{ backgroundColor: '#cccccc', color: 'black' }}
                        className="flex flex-col items-center p-4 h-full">
                        <div className="flex flex-row h-full">
                            <span className="font-bold m-2 text-3xl"
                            >About <br />
                                Us</span>
                            <div className="mt-10 pl-8 w-56 h-44 relative">
                                <img
                                    src="/img/about-us.png"
                                    className="absolute inset-0 w-full h-full object-cover"
                                    alt="Description of the image"
                                />
                                <div className="absolute -top-3 right-0 h-3 w-1/2 bg-black"></div>
                                <div className="absolute bottom-0 -left-3 h-1/2 bg-black w-3"></div>
                            </div>
                        </div>
 
                        <p className="p-4 text-center">
                            <span className="text-wrap">
                                This project is meant to align with two strategic documents from the
                                Melbourne City Council: The Economic Development Strategy, which aims
                                to be a digitally-connected city. The 2021-2025 Council Plan, which
                                outlines the specific objective of delivering programs that will build
                                literacy skills and capabilities.
                            </span>
                        </p>
                    </div>
                    <div
                        style={{ backgroundColor: '#cccccc' ,color: 'black'}}
                        className="flex flex-col items-center p-4 h-full"
                    >
                        <div className="flex flex-row h-full">
                            <span className="font-bold m-2 text-3xl">Open Data Leadership</span>
                            <div className="mt-10 pl-8 w-56 h-44 relative flex-shrink-0">
                                <img
                                    src="/img/leadership.png"
                                    className="absolute inset-0 w-full h-full object-cover"
                                    alt="Description of the image"
                                />
                                <div className="absolute -top-3 right-0 h-3 w-1/2 bg-black"></div>
                                <div className="absolute bottom-0 -left-3 h-1/2 bg-black w-3"></div>
                            </div>
                        </div>
 
                        <p className="p-4 text-center">
                            <span className="text-wrap">
                                This project is meant to align with two strategic documents from the
                                Melbourne City Council: The Economic Development Strategy, which aims
                                to be a digitally-connected city. The 2021-2025 Council Plan, which
                                outlines the specific objective of delivering programs that will build
                                literacy skills and capabilities.
                            </span>
                        </p>
                    </div>
                    <div
                        style={{ backgroundColor: '#cccccc',color: 'black' }}
                        className="flex flex-col items-center p-4 h-full"
                    >
                        <div className="flex flex-row h-full">
                            <span className="font-bold m-2 text-3xl"
                            >Our <br />
                                Goals</span
                            >
                            <div className="mt-10 pl-8 w-56 h-44 relative">
                                <img
                                    src="/img/goals.png"
                                    className="absolute inset-0 w-full h-full object-cover"
                                    alt="Description of the image"
                                />
                                <div className="absolute -top-3 right-0 h-3 w-1/2 bg-black"></div>
                                <div className="absolute bottom-0 -left-3 h-1/2 bg-black w-3"></div>
                            </div>
                        </div>
 
                        <p className="p-4 text-center">
                            <span className="text-wrap">
                                This project is meant to align with two strategic documents from the
                                Melbourne City Council: The Economic Development Strategy, which aims
                                to be a digitally-connected city. The 2021-2025 Council Plan, which
                                outlines the specific objective of delivering programs that will build
                                literacy skills and capabilities.
                            </span>
                        </p>
                    </div>
                </div>
            </div>
 
     <Footer />
 
 
 
        </div>
    
      
    
  );
};

export default About;
