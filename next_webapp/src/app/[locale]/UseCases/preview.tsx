import { useEffect, useState } from "react";

// Define the structure of the use case object
interface Technology {
  name: string;  
  code: string;
  alias: string;
}

interface UseCase {
  _id: string;
  title: string;
  name: string;
  description: string;
  tags: string[];
  difficultly: string;
  technology: Technology[];
  datasets: string[];
}

const PreviewComponent = () => {
  const [caseStudies, setCaseStudies] = useState<UseCase[]>([]);  // Store fetched case studies
  const [selectedCaseStudy, setSelectedCaseStudy] = useState<UseCase | undefined>(undefined);  // Store selected case study
  const [loading, setLoading] = useState<boolean>(true);  // Loading state

  // Fetch case studies from the API when the component mounts
  useEffect(() => {
    const fetchCaseStudies = async () => {
      try {
        const response = await fetch("/api/usecases");  // Fetch case studies from the API
        const data = await response.json();
        setCaseStudies(data);  // Store the fetched case studies
        setSelectedCaseStudy(data[0]);  // Default to the first case study
        setLoading(false);
      } catch (error) {
        console.error("Error fetching case studies:", error);
        setLoading(false);
      }
    };

    fetchCaseStudies();
  }, []);

  if (loading) {
    return <div>Loading...</div>;  // Show a loading message while fetching data
  }

  return (
    <div className="flex h-screen">
      {/* Scrollable Menu on the left */}
      <div className="w-1/4 bg-gray-100 max-h-screen overflow-y-auto">
        <ul>
          {caseStudies.map((study) => (
            <li
              key={study._id}
              className={`text-black mb-2 p-2 hover:bg-gray-200 cursor-pointer ${
                selectedCaseStudy?._id === study._id ? "bg-gray-300" : ""
              }`}
              onClick={() => setSelectedCaseStudy(study)}  // Set the selected case study on click
            >
              {study.title}  {/* Display title in the left panel */}
            </li>
          ))}
        </ul>
      </div>

      {/* Preview Screen on the right */}
      <div className="w-3/4 bg-gray-200 p-4 overflow-y-auto max-h-screen">
        <div className="h-full w-full">
          {/* Display the selected case study */}
          <div className="font-semibold text-2xl">{selectedCaseStudy?.title}</div>
          <p><strong>Name:</strong> {selectedCaseStudy?.name}</p>
          <p><strong>Description:</strong> {selectedCaseStudy?.description}</p>
          <p><strong>Tags:</strong> {selectedCaseStudy?.tags.join(', ')}</p>
          <p><strong>Difficulty:</strong> {selectedCaseStudy?.difficultly}</p>

          {/* Display Technology as a list */}
          <p><strong>Technology:</strong></p>
          <ul>
            {selectedCaseStudy?.technology.map((tech, index) => (
              <li key={index}>
                Name: {tech.name} , Code: {tech.code} {tech.alias ? `(Alias: ${tech.alias})` : ''}
              </li>
            ))}
          </ul>

          {/* Display Datasets as a list */}
          <p><strong>Datasets:</strong></p>
          <ul>
            {selectedCaseStudy?.datasets.map((dataset, index) => (
              <li key={index}>{dataset}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default PreviewComponent;
