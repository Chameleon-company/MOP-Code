// "use client";
// import Header from "../../../components/Header";
// import Footer from "../../../components/Footer";
// import "../../../../public/styles/upload.css";
// import { useEffect, useState, useRef } from "react";
// import "../../../../public/img/Upload_use_case.png";
// import axios from "axios";
// import { TagsInput } from "react-tag-input-component";


// const Upload = () => {
//   const inputRef = useRef<HTMLInputElement>(null);

//   const [isMounted, setIsMounted] = useState(false);
//   const [selectedFile, setSelectedFile] = useState(null);
//   const [selectedFileName, setSelectedFileName] = useState(null);
//   const [progress, setProgress] = useState(0);
//   const [uploadStatus, setUploadStatus] = useState("select");
//   const [tagselect, setTagselect] = useState([" "]);

//   useEffect(() => {
//     setIsMounted(true);
//   }, []);

//   if (!isMounted) {
//     return null;
//   }

//   const handleFileChange = (event) => {
//     if (event.target.files && event.target.files.length > 0) {
//       const file = event.target.files[0];
//       setSelectedFile(file);
//       setSelectedFileName(file.name);
//     }
//   };

//   const onChooseFile = () => {
//     inputRef.current.click();
//   };

//   const clearFileInput = () => {
//     inputRef.current.value = "";
//     setSelectedFile(null);
//     setProgress(0);
//     setUploadStatus("select");
//   };

//   const handleUpload = async () => {
//     if (uploadStatus === "done") {
//       clearFileInput();
//       return;
//     }

//     try {
//       setUploadStatus("uploading");

//       const formData = new FormData();
//       formData.append("file", selectedFile);

//       await axios.post(
//         "http://localhost:3000/en/upload",
//         formData,
//         {
//           onUploadProgress: (progressEvent) => {
//             const percentCompleted = Math.round(
//               (progressEvent.loaded * 100) / progressEvent.total
//             );
//             setProgress(percentCompleted);
//           },
//         }
//       );

//       setUploadStatus("done");
//     } catch (error) {
//       setUploadStatus("select");
//     }
//   };

//   return (
//     <div className="bg-gray-200">
//       <Header />

//       <div className="bg-gray-200  flex justify ">
//         <div className="upload-header-left ">
//           <h1 className="font-bold text-[50px] py-11">{"Upload Case Studies"}</h1>
//         </div>

//       </div>

//       <div className="upload-container">
//         <div className="flex items-center justify-between py-4">
//           <h2 style={{ textAlign: "left" }}>{"Upload Details"}</h2>



//         </div>
//         <div className="form-container">
//           <div className="column">
//             <label htmlFor="Name">{"Name"}</label>
//             <input type="text" id="first-name" name="first-name" placeholder={"Enter  name"} />


//             {/* <pre>{JSON.stringify(tagselect)}</pre> */}
//             <label htmlFor="Tag">{"Tags"}</label>
//             <TagsInput
//               value={tagselect}
//               onChange={setTagselect}
//               name="tags"
//               placeHolder="tags"
//             />
//           </div>
//           <div className="column">

//             <label htmlFor="description" className="ml-5">{"Description"}</label>
//             <input type="text" id="last-name" className="ml-5" name="last-name" placeholder={"Enter Description"} />

//             <div className="column m-0">
//               <label htmlFor="description" className=" text-lg font-medium text-gray-700">{"Trimester"}</label>
//             </div>
//             <div className="column m-0">
//               <select className="border border-gray-300 rounded-md px-[93%] py-3  my-1"
//                 name="trimester"
//                 id="trimester" >
//                 <option value="option1">{"Trimester 1"}</option>
//                 <option value="option2">{"Trimester 2"}</option>
//                 <option value="option3">{"Trimester 3"}</option>
//               </select>

//             </div>

//           </div>
//         </div>

//         <div className="flex justify-center">
//           <div className="border-dashed border-2 border-black w-[50rem] h-[25rem] items-center">
//             <div className="flex items-center justify-center pt-[8rem]">
//               <input ref={inputRef} type="file" onChange={handleFileChange} style={{ display: "none" }} />
//               <button onClick={onChooseFile}>
//                 <img className="h-20 w-auto" src="../img/Upload_use_case.png" alt="Logo" />
//               </button>
//             </div>
//             <h1 className="text-center text-lg py-[2rem]">{"Click on logo to upload files"}</h1>
//           </div>
//         </div>

//         {selectedFile && (
//           <>
//             <div className="flex pt-8 justify-center items-center">
//               <div className="bg-gray-200 w-[50rem] py-[3rem] px-2 rounded-3xl">
//                 <div className="flex">
//                   <img className="flex-initial h-8 w-auto" src="../img/document.png" alt="Document Icon" />
//                   <h6 className="flex-1 font-bold items-center pl-2">{selectedFileName}</h6>
//                   <p> {progress}%</p>
//                 </div>
//                 <div className="w-full pt-3 bg-gray-200 rounded-full h-2.5 dark:bg-gray-700 pl-2">
//                   <div className="bg-green-600 h-2.5 rounded-full" style={{ width: `${progress}%` }}></div>
//                 </div>
//               </div>
//             </div>

//             <div className="flex justify-center pt-4">
//               <button
//                 className="bg-green-500 text-white px-4 py-2 rounded"
//                 onClick={handleUpload}
//               >
//                 {uploadStatus === "done" ? "Clear" : "Upload File"}
//               </button>
//             </div>
//           </>
//         )}
//       </div>

//       <div className="spacer"></div>
//       <Footer />
//     </div >
//   );
// };

// export default Upload;

//Divyanga C.S.Lokuhetti
//s223590519
//Team Project(B)
//T1 2025

"use client";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { useEffect, useState, useRef } from "react";
import { TagsInput } from "react-tag-input-component";
import axios from "axios";
import Tooglebutton from "../Tooglebutton/Tooglebutton";

const Upload = () => {
  const inputRef = useRef<HTMLInputElement>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [uploadStatus, setUploadStatus] = useState<"select" | "uploading" | "done">("select");
  const [tagselect, setTagselect] = useState<string[]>([]);
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    const theme = localStorage.getItem("theme");
    if (theme === "dark") {
      setDarkMode(true);
      document.documentElement.classList.add("dark");
    }
  }, []);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [darkMode]);

  const handleToggle = (val: boolean) => {
    setDarkMode(val);
    localStorage.setItem("theme", val ? "dark" : "light");
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setSelectedFileName(file.name);
    }
  };

  const onChooseFile = () => {
    inputRef.current?.click();
  };

  const clearFileInput = () => {
    if (inputRef.current) inputRef.current.value = "";
    setSelectedFile(null);
    setSelectedFileName(null);
    setProgress(0);
    setUploadStatus("select");
  };

  const handleUpload = async () => {
    if (uploadStatus === "done") {
      clearFileInput();
      return;
    }

    if (!selectedFile) return;

    try {
      setUploadStatus("uploading");

      const formData = new FormData();
      formData.append("file", selectedFile);

      await axios.post("/api/upload", formData, {
        onUploadProgress: (event) => {
          const percent = Math.round((event.loaded * 100) / (event.total ?? 1));
          setProgress(percent);
        },
      });

      setUploadStatus("done");
    } catch (err) {
      console.error("Upload failed", err);
      setUploadStatus("select");
    }
  };

  return (
    <div className="bg-gray-100 dark:bg-[#1d1919] min-h-screen text-black dark:text-white transition-all duration-300">
      <Header />

      <main className="px-8 py-10 font-sans max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold mb-10">Upload Case Studies</h1>

        <div className="bg-white dark:bg-[#2a2a2a] rounded-xl shadow-md p-8">
          <h2 className="text-2xl font-semibold mb-6">Upload Details</h2>

          <form className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block mb-2">Name</label>
              <input
                type="text"
                placeholder="Enter name"
                className="w-full p-3 rounded-md border border-gray-300 dark:border-gray-600 dark:bg-[#1d1d1d] dark:text-white"
              />

              <label className="block mt-6 mb-2">Tags</label>
              <TagsInput
                value={tagselect}
                onChange={setTagselect}
                name="tags"
                placeHolder="tags"
                classNames={{
                  input:
                    "dark:bg-[#1d1d1d] dark:text-white border border-gray-300 dark:border-gray-600 rounded-md p-2",
                  tag: "bg-green-500 text-white px-2 py-1 rounded",
                  tagRemove: "ml-1 cursor-pointer",
                }}
              />
            </div>

            <div>
              <label className="block mb-2">Description</label>
              <input
                type="text"
                placeholder="Enter description"
                className="w-full p-3 rounded-md border border-gray-300 dark:border-gray-600 dark:bg-[#1d1d1d] dark:text-white"
              />

              <label className="block mt-6 mb-2">Trimester</label>
              <select
                className="w-full p-3 rounded-md border border-gray-300 dark:border-gray-600 dark:bg-[#1d1d1d] dark:text-white"
              >
                <option>Trimester 1</option>
                <option>Trimester 2</option>
                <option>Trimester 3</option>
              </select>
            </div>
          </form>

          {/* File upload area */}
          <div className="mt-10 border-2 border-dashed border-gray-400 dark:border-gray-600 rounded-md p-10 text-center">
            <input ref={inputRef} type="file" onChange={handleFileChange} hidden />
            <button type="button" onClick={onChooseFile}>
              <img
                src="/img/Upload_use_case.png"
                alt="Upload"
                className="h-20 w-auto mx-auto"
              />
            </button>
            <p className="text-lg mt-4">Click on logo to upload files</p>
          </div>

          {selectedFile && (
            <>
              <div className="mt-8 bg-gray-100 dark:bg-[#1a1a1a] p-6 rounded-lg shadow-inner">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <img src="/img/document.png" alt="doc" className="h-6" />
                    <span>{selectedFileName}</span>
                  </div>
                  <span>{progress}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 h-2 rounded-full mt-4">
                  <div
                    className="bg-green-500 h-2 rounded-full"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>

              <div className="mt-6 text-center">
                <button
                  className="bg-green-600 text-white px-6 py-2 rounded hover:bg-green-700"
                  onClick={handleUpload}
                >
                  {uploadStatus === "done" ? "Clear" : "Upload File"}
                </button>
              </div>
            </>
          )}
        </div>
      </main>

      {/* Dark mode toggle */}
      <div className="fixed bottom-4 right-4 z-50">
        <Tooglebutton onValueChange={handleToggle} />
      </div>

      <Footer />
    </div>
  );
};

export default Upload;
