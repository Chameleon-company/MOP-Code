"use client";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/upload.css";
import { useEffect, useState, useRef } from "react";
import "../../../../public/img/Upload_use_case.png";
import axios from "axios";
import { TagsInput } from "react-tag-input-component";


const Upload = () => {
  const inputRef = useRef<HTMLInputElement>(null);

  const [isMounted, setIsMounted] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedFileName, setSelectedFileName] = useState(null);
  const [progress, setProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState("select");
  const [tagselect, setTagselect] = useState([" "]);
  
  //handlers for upload
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) {
    return null;
  }

  const handleFileChange = (event) => {
    if (event.target.files && event.target.files.length > 0) {
      const file = event.target.files[0];
      setSelectedFile(file);
      setSelectedFileName(file.name);
    }
  };

  const onChooseFile = () => {
    inputRef.current.click();
  };

  const clearFileInput = () => {
    inputRef.current.value = "";
    setSelectedFile(null);
    setProgress(0);
    setUploadStatus("select");
  };

  const handleUpload = async () => {
    if (!title || !description) {
      alert('Title and description cannot be null!');
    }
    else {
      if (uploadStatus === "done") {
        clearFileInput();
        return;
      }

      try {
        setUploadStatus("uploading");
  
        const formData = new FormData();
        formData.append("name", title);
        formData.append("description", description);
        formData.append("tags", tagselect);
        formData.append("file", selectedFile);
  
        const response = await axios.post(
          // "http://localhost:3000/en/upload",
          "/api/usecases",
          {title, description, tagselect, selectedFile},
          {
            onUploadProgress: (progressEvent) => {
              const percentCompleted = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );
              setProgress(percentCompleted);
            },
          }
        );
        console.log('test1');
        setUploadStatus("done");
      } catch (error) {
        console.log('test2');
        setUploadStatus("select");
      }
    }    
  };

  return (
    <div className="bg-gray-200">
      <Header />

      <div className="bg-gray-200  flex justify ">
        <div className="upload-header-left ">
          <h1 className="font-bold text-[50px] py-11">{"Upload Case Studies"}</h1>
        </div>

      </div>

      <div className="upload-container">
        <div className="flex items-center justify-between py-4">
          <h2 style={{ textAlign: "left" }}>{"Upload Details"}</h2>



        </div>
        <div className="form-container">
          <div className="column">
            <label htmlFor="Name">{"Name"}</label>
            <input
              onChange={(e) => setTitle(e.target.value)}
              type="text"
              id="title"
              name="title"
              placeholder={"Enter title"}
            />

            {/* <pre>{JSON.stringify(tagselect)}</pre> */}
            <label htmlFor="Tag">{"Tags"}</label>
            <TagsInput
              value={tagselect}
              onChange={setTagselect}
              name="tags"
              placeHolder="tags"
            />
          </div>
          <div className="column">

            <label htmlFor="description" className="ml-5">{"Description"}</label>
            <input
              onChange={(e) => setDescription(e.target.value)}
              type="text"
              id="last-name"
              className="ml-5"
              name="last-name"
              placeholder={"Enter Description"}
            />

            <div className="column m-0">
              <label htmlFor="description" className=" text-lg font-medium text-gray-700">{"Trimester"}</label>
            </div>
            <div className="column m-0">
              <select className="border border-gray-300 rounded-md px-[93%] py-3  my-1"
                name="trimester"
                id="trimester" >
                <option value="option1">{"Trimester 1"}</option>
                <option value="option2">{"Trimester 2"}</option>
                <option value="option3">{"Trimester 3"}</option>
              </select>

            </div>

          </div>
        </div>

        <div className="flex justify-center">
          <div className="border-dashed border-2 border-black w-[50rem] h-[25rem] items-center">
            <div className="flex items-center justify-center pt-[8rem]">
              <input ref={inputRef} type="file" onChange={handleFileChange} style={{ display: "none" }} />
              <button onClick={onChooseFile}>
                <img className="h-20 w-auto" src="../img/Upload_use_case.png" alt="Logo" />
              </button>
            </div>
            <h1 className="text-center text-lg py-[2rem]">{"Click on logo to upload files"}</h1>
          </div>
        </div>

        {selectedFile && (
          <>
            <div className="flex pt-8 justify-center items-center">
              <div className="bg-gray-200 w-[50rem] py-[3rem] px-2 rounded-3xl">
                <div className="flex">
                  <img className="flex-initial h-8 w-auto" src="../img/document.png" alt="Document Icon" />
                  <h6 className="flex-1 font-bold items-center pl-2">{selectedFileName}</h6>
                  <p> {progress}%</p>
                </div>
                <div className="w-full pt-3 bg-gray-200 rounded-full h-2.5 dark:bg-gray-700 pl-2">
                  <div className="bg-green-600 h-2.5 rounded-full" style={{ width: `${progress}%` }}></div>
                </div>
              </div>
            </div>
          </>
        )}

        <div className="flex justify-center pt-4">
                <button
                  className="bg-green-500 text-white px-4 py-2 rounded"
                  onClick={handleUpload}
                >
                  {uploadStatus === "done" ? "Clear" : "Upload File"}
                </button>
              </div>
        </div>

      <div className="spacer"></div>
      <Footer />
    </div >
  );
};

export default Upload;
