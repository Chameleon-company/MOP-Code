"use client";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/upload.css";
import { useTranslations } from "next-intl";
import { useEffect, useState, useRef } from "react";
import "../../../../public/img/Upload_use_case.png";
import axios from "axios";
import { TagsInput } from "react-tag-input-component";


const Upload = () => {
  const t = useTranslations("upload");
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
    setUploadStatus(t("select"));
  };

  const handleUpload = async () => {
    if (uploadStatus === t("done")) {
      clearFileInput();
      return;
    }

    try {
      setUploadStatus(t("uploading"));

      const formData = new FormData();
      formData.append(t("file"), selectedFile);

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

      setUploadStatus(t("done"));
    } catch (error) {
      setUploadStatus(t("select"));
    }
  };

  return (
    <div className="bg-gray-200 dark:bg-[#263238] dark:text-white">
      <Header />

      <div className="bg-gray-200 text-center">
          <h1 className="font-bold text-2xl sm:text-4xl py-11">{"Upload Case Studies"}</h1>
      </div>

      <div className="upload-container bg-white dark:bg-[#263238] dark:text-white">
        <div className="text-center sm:text-left sm:text-xl sm:mb-2">
          <h3>{"Upload Details"}</h3>
        </div>
        <div className="form-container">
          <div className="column">
            <label htmlFor="Name">{t("Name")}</label>
            <input className='dark:text-[#263238]' type="text" id="first-name" name="first-name" placeholder={"Enter  name"} />


            {/* <pre>{JSON.stringify(tagselect)}</pre> */}
            <label htmlFor="Tag">{"Tags"}</label>
            <div className="taginput dark:text-[#263238]">
            <TagsInput
              value={tagselect}
              onChange={setTagselect}
              name="tags"
              placeHolder={t("Tags")}
            />

            </div>
          </div>
          <div className="column">

            <label htmlFor="description">{"Description"}</label>
            <input className="dark:text-[#263238]" type="text" id="last-name"  name="last-name" placeholder={"Enter Description"} />

            
              <label htmlFor="trimester">{"Trimester"}</label>

            <select className=" border border-gray-300 rounded-md py-3  my-1 dark:text-[#263238]"
                name="trimester"
                id="trimester" >
                <option value="option1">{t("Trimester 1")}</option>
                <option value="option2">{t("Trimester 2")}</option>
                <option value="option3">{t("Trimester 3")}</option>
              </select>
              

            

          </div>
        </div>

        <div className="flex justify-center">
          <div className="border-dashed border-2 border-black dark:border-white w-[50rem] h-[25rem] items-center">
            <div className="flex items-center justify-center pt-[8rem]">
              <input ref={inputRef} type="file" onChange={handleFileChange} style={{ display: "none" }} />
              <button onClick={onChooseFile}>
                <img className="h-20 w-auto" src="../img/Upload_use_case.png" alt="Logo" />
              </button>
            </div>
            <h1 className="text-center text-lg py-[2rem]">{t("Click on logo to upload files")}</h1>
          </div>
        </div>

        {selectedFile && (
          <>
            <div className="flex pt-8 justify-center items-center">
              <div className="bg-gray-200 w-[50rem] py-[3rem] px-2 rounded-3xl">
                <div className="flex">
                  <img className="flex-initial h-8 w-auto" src="../img/document.png" alt="Document Icon" />
                  <h6 className="flex-1 font-bold items-center pl-2 dark:text-[#263238]">{selectedFileName}</h6>
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
                {uploadStatus === "done" ? t("Clear") : t("Upload File")}
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
