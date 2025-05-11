"use client";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/upload.css";
import { useTranslations } from "next-intl";
import { useEffect, useState, useRef } from "react";
import { TagsInput } from "react-tag-input-component";
import axios from "axios";
import Tooglebutton from "../Tooglebutton/Tooglebutton";

const Upload = () => {
  const t = useTranslations("upload");
  const inputRef = useRef<HTMLInputElement>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [uploadStatus, setUploadStatus] = useState<
    "select" | "uploading" | "done"
  >("select");
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
    } catch (error) {
      console.error("Upload failed", error);
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
              <select className="w-full p-3 rounded-md border border-gray-300 dark:border-gray-600 dark:bg-[#1d1d1d] dark:text-white">
                <option>Trimester 1</option>
                <option>Trimester 2</option>
                <option>Trimester 3</option>
              </select>
            </div>
          </form>

          <div className="mt-10 border-2 border-dashed border-gray-400 dark:border-gray-600 rounded-md p-10 text-center">
            <input
              ref={inputRef}
              type="file"
              onChange={handleFileChange}
              hidden
            />
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
