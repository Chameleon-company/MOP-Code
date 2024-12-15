// import required libraries
import { CATEGORY, CaseStudy, SEARCH_MODE, SearchParams } from "@/app/types";
import fs from "fs";
import path from "path";

// defaults to auto
// Create a Server Component
export async function POST(request: Request) {
  const baseDir = path.join(
    process.cwd(),
    "..",
    "datascience",
    "usecases",
    "READY TO PUBLISH"
  );
  const res = (await request.json()) as SearchParams;
  console.log("🚀 ~ POST ~ resn:", res);
  const { category, searchMode, searchTerm } = res;

  const caseStudies: CaseStudy[] = [];
  let id = 1;

  // Get all subdirectories in the base directory (e.g., T1 2024, T2 2024, etc.)
  const termFolders = fs
    .readdirSync(baseDir, { withFileTypes: true })
    .filter((dirent) => dirent.isDirectory())
    .map((dirent) => dirent.name);

  termFolders.forEach((termFolder) => {

    const termFolderPath = path.join(baseDir, termFolder);

    // Get all subdirectories
    const subdirs = fs
      .readdirSync(termFolderPath, { withFileTypes: true })
      .filter((dirent) => dirent.isDirectory())
      .map((dirent) => dirent.name);

    subdirs.forEach(subdir => {
      const subdirPath = path.join(termFolderPath, subdir);
      const files = fs.readdirSync(subdirPath);

      // Find JSON and HTML files
      const jsonFile = files.find(file => file.endsWith('.json'));
      const htmlFile = files.find(file => file.endsWith('.html'));

      try {
        if (jsonFile && htmlFile) {
          const jsonPath = path.join(subdirPath, jsonFile);
          const jsonContent = fs.readFileSync(jsonPath, 'utf-8');
          const { name, description, tags } = JSON.parse(jsonContent);

          caseStudies.push({
            id: id++,
            description: description,
            name: name,
            tags: tags,
            filename: path.join(baseDir,termFolder, subdir, htmlFile)
          });
        }
      }
      catch (e) {
        console.log("Invalid UseCase Format: ", subdirPath, e)
      }
    });
  });

  let filteredStudies: (CaseStudy & { fileContent?: string })[] = [];
  if (searchMode === SEARCH_MODE.TITLE) {

    filteredStudies = caseStudies.filter((caseStudy) => {
      return (
        caseStudy.name.toLowerCase().includes(searchTerm.toLowerCase())
      );
    });
  } else if (searchMode === SEARCH_MODE.CONTENT) {
    filteredStudies = caseStudies.filter((caseStudy) => {
      return caseStudy.tags.some((tag) =>
        tag.toLowerCase().includes(searchTerm.toLowerCase())
      );
    });
  }

  console.log("🚀 ~ POST ~ filteredStudies:", filteredStudies);



  return Response.json({ filteredStudies });

  // Extracting Filename from Query Parameters
  // const { searchParams } = new URL(request.url)
  // const fileName = `${searchParams.get('filename')}.html`
  // console.log(fileName)

  // // Constructing the File Path
  // const filePath = path.join(
  //   process.cwd(),
  //   "public",
  //   fileName
  // );
  // // Reading the file
  // const file = await fs.readFileSync(filePath, "utf-8");

  // Parsing the File's Contents and Returning the Response
  // return new Response(file, {
  //   status: 200,
  //   headers: {
  //     "Content-Type": "text/html; charset=utf-8",
  //   },
  // });
}