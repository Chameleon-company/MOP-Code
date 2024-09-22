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
    <div className=" dark:bg-zinc-800">
      <Header />
      <main className="upload-body dark:bg-zinc-800">
        <div className="upload-header  dark:bg-zinc-800 dark:text-slate-100">
          <div className="upload-header-left">
            <h1>{"Upload Case Studies"}</h1>
          </div>
          <div className="upload-header-right">
            <select style={{ border: "none" }} className=" dark:bg-zinc-800">
              <option value="option1">{"Trimester 1"}</option>
              <option value="option2">{"Trimester 2"}</option>
              <option value="option3">{"Trimester 3"}</option>
            </select>
          </div>
        </div>
        <div className="upload-container dark:bg-zinc-700 dark:text-slate-100">

          <h2 style={{ textAlign: "left" }}>{"Uploader's Details"}</h2>
          <div className="form-container">
            <div className="column">
              <label htmlFor="first-name">{"Author's Name"}</label>
              <input type="text" id="first-name" className="dark:bg-zinc-700" name="first-name" placeholder={"Enter author's name"} />

              <label htmlFor="last-name">{"DOP"}</label>
              <input type="text" id="last-name" className="dark:bg-zinc-700" name="last-name" placeholder={"Enter DOP"} />

              <label htmlFor="email">{"Company Email"}</label>
              <input type="email" id="email" name="email" className="dark:bg-zinc-700" placeholder={"Enter email"} />
            </div>

            <div className="column">
              <label htmlFor="phone">{"Case Study"}</label>
              <input type="tel" id="phone" name="phone" className="dark:bg-zinc-700" placeholder={"Enter case "} />

              <label htmlFor="address">{"Category"}</label>
              <input type="text" id="address" name="address" className="dark:bg-zinc-700" placeholder={"Enter category"} />
            </div>
          </div>
          <div className="green-box">
            <svg width="800px" height="800px" viewBox="0 0 1024 1024" className="upload-icon" version="1.1" xmlns="http://www.w3.org/2000/svg">
              <path d="M736.68 435.86a173.773 173.773 0 0 1 172.042 172.038c0.578 44.907-18.093 87.822-48.461 119.698-32.761 34.387-76.991 51.744-123.581 52.343-68.202 0.876-68.284 106.718 0 105.841 152.654-1.964 275.918-125.229 277.883-277.883 1.964-152.664-128.188-275.956-277.883-277.879-68.284-0.878-68.202 104.965 0 105.842zM285.262 779.307A173.773 173.773 0 0 1 113.22 607.266c-0.577-44.909 18.09-87.823 48.461-119.705 32.759-34.386 76.988-51.737 123.58-52.337 68.2-0.877 68.284-106.721 0-105.842C132.605 331.344 9.341 454.607 7.379 607.266 5.417 759.929 135.565 883.225 285.262 885.148c68.284 0.876 68.2-104.965 0-105.841z" fill="#4A5699" />
              <path d="M339.68 384.204a173.762 173.762 0 0 1 172.037-172.038c44.908-0.577 87.822 18.092 119.698 48.462 34.388 32.759 51.743 76.985 52.343 123.576 0.877 68.199 106.72 68.284 105.843 0-1.964-152.653-125.231-275.917-277.884-277.879-152.664-1.962-275.954 128.182-277.878 277.879-0.88 68.284 104.964 68.199 105.841 0z" fill="#C45FA0" />
              <path d="M545.039 473.078c16.542 16.542 16.542 43.356 0 59.896l-122.89 122.895c-16.542 16.538-43.357 16.538-59.896 0-16.542-16.546-16.542-43.362 0-59.899l122.892-122.892c16.537-16.542 43.355-16.542 59.894 0z" fill="#F39A2B" />
              <path d="M485.17 473.078c16.537-16.539 43.354-16.539 59.892 0l122.896 122.896c16.538 16.533 16.538 43.354 0 59.896-16.541 16.538-43.361 16.538-59.898 0L485.17 532.979c-16.547-16.543-16.547-43.359 0-59.901z" fill="#F39A2B" />
              <path d="M514.045 634.097c23.972 0 43.402 19.433 43.402 43.399v178.086c0 23.968-19.432 43.398-43.402 43.398-23.964 0-43.396-19.432-43.396-43.398V677.496c0.001-23.968 19.433-43.399 43.396-43.399z" fill="#E5594F" />
            </svg>
            <div className="file-upload">
              <h1 className="dark:text-black">{"Drag & Drop Files"}</h1>
              <input type="file" id="file-upload" className="file-upload-input" />
              <button className="upload-btn">{"UPLOAD FILES"}</button>
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
            <input type="text" id="first-name" name="first-name" placeholder={"Enter  name"} />


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
            <input type="text" id="last-name" className="ml-5" name="last-name" placeholder={"Enter Description"} />

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

            <div className="flex justify-center pt-4">
              <button
                className="bg-green-500 text-white px-4 py-2 rounded"
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
    </div >
  );
};

export default Upload;
