// import required libraries
import { caseStudies } from "@/app/[locale]/UseCases/database";
import { CATEGORY, CaseStudy, SEARCH_MODE, SearchParams } from "@/app/types";
import fs from "fs";
import path from "path";

// defaults to auto
// Create a Server Component
export async function POST(request: Request) {
  const res = (await request.json()) as SearchParams;
  console.log("🚀 ~ POST ~ res:", res);
  const { category, searchMode, searchTerm } = res;

  let filteredStudies: (CaseStudy & { fileContent?: string })[] = [];
  if (searchMode === SEARCH_MODE.TITLE) {
    filteredStudies = caseStudies.filter((caseStudy) => {
      return (
        (category === CATEGORY.ALL || category === caseStudy.category) &&
        caseStudy.title.toLowerCase().includes(searchTerm.toLowerCase())
      );
    });
  } else if (searchMode === SEARCH_MODE.CONTENT) {
    const caseStudiesWithContent = await Promise.all(
      caseStudies.map(async (study) => {
        const filePath = path.join(
          process.cwd(),
          "public",
          `${study.filename}.html`
        ); // Ensure the file extension is correct
        try {
          const fileContent = await fs.readFileSync(filePath, "utf-8");
          return { ...study, fileContent }; // Spread the existing study object and add the file content
        } catch (error) {
          console.error(
            `Failed to read file for Case Study ${study.id}: ${error}`
          );
          return { ...study, fileContent: "Failed to load file content" }; // Handle errors
        }
      })
    );
    filteredStudies = caseStudiesWithContent.filter((caseStudy) => {
      return (
        (category === CATEGORY.ALL || category === caseStudy.category) &&
        caseStudy.fileContent?.toLowerCase().includes(searchTerm.toLowerCase())
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
