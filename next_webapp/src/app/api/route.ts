// import required libraries
import fs from "fs";
import path from "path";

// defaults to auto
// Create a Server Component
export async function GET(request: Request) {

// Extracting Filename from Query Parameters
  const { searchParams } = new URL(request.url)
  const fileName = `${searchParams.get('filename')}.html`
  console.log(fileName)

  // Constructing the File Path
  const filePath = path.join(
    process.cwd(),
    "public",
    fileName
  );
  // Reading the file
  const file = await fs.readFileSync(filePath, "utf-8");


  // Parsing the File's Contents and Returning the Response
  return new Response(file, {
    status: 200,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
    },
  });
}
