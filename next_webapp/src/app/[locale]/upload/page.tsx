"use client";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/upload.css";
import { useEffect, useState, useRef } from "react";
import "../../../../public/img/Upload_use_case.png";
import axios from "axios";

const Upload = () => {
  const inputRef = useRef<HTMLInputElement>(null);

  const [isMounted, setIsMounted] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedFileName, setSelectedFileName] = useState(null);
  const [progress, setProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState("select");

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
    if (uploadStatus === "done") {
      clearFileInput();
      return;
    }

    try {
      setUploadStatus("uploading");

      const formData = new FormData();
      formData.append("file", selectedFile);

      await axios.post(
        "http://localhost:3000/en/upload",
        formData,
        {
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setProgress(percentCompleted);
          },
        }
      );

      setUploadStatus("done");
    } catch (error) {
      setUploadStatus("select");
    }
  };

  return (
    <div className="bg-gray-200">
      <Header />

      <div className="bg-gray-200  flex justify ">
        <div className="upload-header-left ">
          <h1 className="font-bold text-[50px] py-11">{"Upload Case Studies"}</h1>
        </div>
        <div className="  ml-[80rem] mt-20 py-15">
          <select className=" py-15 p-12px" style={{ border: "none" }}>
            <option value="option1">{"Trimester 1"}</option>
            <option value="option2">{"Trimester 2"}</option>
            <option value="option3">{"Trimester 3"}</option>
          </select>
        </div>
      </div>

      <div className="upload-container">
        <h2 style={{ textAlign: "left" }}>{"Uploader's Details"}</h2>
        <div className="form-container">
          <div className="column">
            <label htmlFor="first-name">{"Author's Name"}</label>
            <input type="text" id="first-name" name="first-name" placeholder={"Enter author's name"} />

            <label htmlFor="last-name">{"DOP"}</label>
            <input type="text" id="last-name" name="last-name" placeholder={"Enter DOP"} />

            <label htmlFor="email">{"Company Email"}</label>
            <input type="email" id="email" name="email" placeholder={"Enter email"} />
          </div>

          <div className="column">
            <label htmlFor="phone">{"Case Study"}</label>
            <input type="tel" id="phone" name="phone" placeholder={"Enter case "} />

            <label htmlFor="address">{"Category"}</label>
            <input type="text" id="address" name="address" placeholder={"Enter category"} />
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
            <h1 className="text-center text-lg py-[2rem]">{"Click on Upload logo "}</h1>
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
                  <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${progress}%` }}></div>
                </div>
              </div>
            </div>

            <div className="flex justify-center pt-4">
              <button
                className="bg-blue-500 text-white px-4 py-2 rounded"
                onClick={handleUpload}
              >
                {uploadStatus === "done" ? "Clear" : "Upload File"}
              </button>
            </div>
          </>
        )}
      </div>

      <div className="spacer"></div>
      <Footer />
    </div>
  );
};

export default Upload;
