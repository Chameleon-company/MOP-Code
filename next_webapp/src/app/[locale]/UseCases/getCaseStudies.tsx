const fs = require('fs');
const path = require('path');

// Define a function to get case studies from folders
const getCaseStudiesFromFolders = (baseDir) => {
  const caseStudies = [];

  // Read all folders in the base directory
  const folders = fs.readdirSync(baseDir);

  folders.forEach((folder) => {
    const folderPath = path.join(baseDir, folder);
    
    if (fs.lstatSync(folderPath).isDirectory()) {
      // Read the contents of the folder
      const files = fs.readdirSync(folderPath);

      // Find the JSON file
      const jsonFile = files.find(file => file.endsWith('.json'));

      if (jsonFile) {
        const jsonFilePath = path.join(folderPath, jsonFile);
        const jsonData = JSON.parse(fs.readFileSync(jsonFilePath, 'utf-8'));

        // Extract required data
        const title = jsonData.title || folder;
        const filename = jsonData.filename || `${folder}.html`;

        // Create a case study object
        const caseStudy = {
          id: caseStudies.length + 1,  // Automatically increment ID
          title,
          content: jsonData.content || 'No content provided.',
          category: jsonData.category || 'DEFAULT',
          filename
        };

        caseStudies.push(caseStudy);
      }
    }
  });

  return caseStudies;
};

// Usage
const baseDir = '/path/to/your/folders';
const caseStudies = getCaseStudiesFromFolders(baseDir);

// Now caseStudies array can be used in your PreviewComponent
