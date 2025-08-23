"use client";

import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import { useTranslations } from "next-intl";
import { useEffect, useRef, useState } from "react";
import { TagsInput } from "react-tag-input-component";
import axios from "axios";

// Client-side guards to match backend rules
const ALLOWED_TYPES = [
  "text/csv",
  "application/vnd.ms-excel",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/json",
];
const MAX_SIZE_MB = 20;

export default function Upload() {
  const t = useTranslations("upload");
  const inputRef = useRef<HTMLInputElement>(null);

  // form state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [trimester, setTrimester] = useState("Trimester 1"); // optional metadata

  // ui state
  const [progress, setProgress] = useState<number>(0);
  const [uploadStatus, setUploadStatus] =
    useState<"select" | "uploading" | "done">("select");
  const [darkMode, setDarkMode] = useState(false);

  // load persisted theme
  useEffect(() => {
    const theme = localStorage.getItem("theme");
    if (theme === "dark") {
      setDarkMode(true);
      document.documentElement.classList.add("dark");
    }
  }, []);
  useEffect(() => {
    if (darkMode) document.documentElement.classList.add("dark");
    else document.documentElement.classList.remove("dark");
  }, [darkMode]);

  const onChooseFile = () => inputRef.current?.click();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setSelectedFileName(file.name);
    }
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
    if (!selectedFile || !name || !email || !title || !description) {
      alert("Please fill Name, Email, Title, Description and choose a file.");
      return;
    }
    if (!ALLOWED_TYPES.includes(selectedFile.type)) {
      alert("Unsupported file type. Allowed: CSV, XLS, XLSX, JSON.");
      return;
    }
    if (selectedFile.size > MAX_SIZE_MB * 1024 * 1024) {
      alert(`File too large. Max ${MAX_SIZE_MB} MB.`);
      return;
    }

    try {
      setUploadStatus("uploading");
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("name", name);
      formData.append("email", email);
      formData.append("title", title);
      formData.append("description", description);
      formData.append("tags", tags.join(",")); // backend splits by comma
      formData.append("trimester", trimester); // optional

      await axios.post("/api/contributions", formData, {
        onUploadProgress: (event) => {
          const total = event.total ?? 1;
          const percent = Math.round((event.loaded * 100) / total);
          setProgress(percent);
        },
      });

      setUploadStatus("done");
    } catch (err) {
      console.error("Upload failed", err);
      setUploadStatus("select");
      setProgress(0);
      alert("Upload failed. Please try again.");
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 text-black transition-all duration-300 dark:bg-[#1d1919] dark:text-white">
      <Header />

      <main className="mx-auto max-w-6xl px-8 py-10 font-sans">
        <h1 className="mb-10 text-4xl font-bold">{t("Upload Case Studies")}</h1>

        <div className="rounded-xl bg-white p-8 shadow-md dark:bg-[#2a2a2a]">
          <h2 className="mb-6 text-2xl font-semibold">{t("Upload Details")}</h2>

          <form className="mb-10 grid grid-cols-1 gap-6 md:grid-cols-2" onSubmit={(e) => e.preventDefault()}>
            <div>
              <label className="mb-2 block">{t("Name")}</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter name"
                className="w-full rounded-md border border-gray-300 p-3 dark:border-gray-600 dark:bg-[#1d1d1d] dark:text-white"
              />

              <label className="mt-6 mb-2 block">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter email"
                className="w-full rounded-md border border-gray-300 p-3 dark:border-gray-600 dark:bg-[#1d1d1d] dark:text-white"
              />

              <label className="mt-6 mb-2 block">{t("Tags")}</label>
              <TagsInput
                value={tags}
                onChange={setTags}
                name="tags"
                placeHolder="Tags"
                classNames={{
                  input:
                    "rounded-md border border-gray-300 p-2 dark:border-gray-600 dark:bg-[#1d1d1d] dark:text-white",
                  tag: "rounded bg-green-500 px-2 py-1 text-white",
                }}
              />
            </div>

            <div>
              <label className="mb-2 block">Title</label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Dataset title"
                className="w-full rounded-md border border-gray-300 p-3 dark:border-gray-600 dark:bg-[#1d1d1d] dark:text-white"
              />

              <label className="mt-6 mb-2 block">{t("Description")}</label>
              <input
                type="text"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Enter description"
                className="w-full rounded-md border border-gray-300 p-3 dark:border-gray-600 dark:bg-[#1d1d1d] dark:text-white"
              />

              <label className="mt-6 mb-2 block">{t("Trimester")}</label>
              <select
                value={trimester}
                onChange={(e) => setTrimester(e.target.value)}
                className="w-full rounded-md border border-gray-300 p-3 dark:border-gray-600 dark:bg-[#1d1d1d] dark:text-white"
              >
                <option>{t("Trimester 1")}</option>
                <option>{t("Trimester 2")}</option>
                <option>{t("Trimester 3")}</option>
              </select>
            </div>
          </form>

          <div className="rounded-md border-2 border-dashed border-gray-400 p-10 text-center dark:border-gray-600">
            <input
              ref={inputRef}
              type="file"
              onChange={handleFileChange}
              hidden
              accept=".csv,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/json"
            />
            <button type="button" onClick={onChooseFile}>
              <img src="/img/Upload_use_case.png" alt="Upload" className="mx-auto h-20 w-auto" />
            </button>
            <p className="mt-4 text-lg">{t("Click on logo to upload files")}</p>
          </div>

          {selectedFile && (
            <div className="mt-8 rounded-lg bg-gray-100 p-6 shadow-inner dark:bg-[#1a1a1a]">
              <div className="mb-2 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <img src="/img/document.png" alt="doc" className="h-6" />
                  <span>{selectedFileName}</span>
                </div>
                <span>{progress}%</span>
              </div>
              <div className="h-2 w-full rounded-full bg-gray-200 dark:bg-gray-700">
                <div className="h-2 rounded-full bg-green-500" style={{ width: `${progress}%` }} />
              </div>

              <div className="mt-6 text-center">
                <button
                  className="rounded bg-green-600 px-6 py-2 text-white hover:bg-green-700"
                  onClick={handleUpload}
                >
                  {uploadStatus === "done" ? t("Clear") : t("Upload File")}
                </button>
              </div>
            </div>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
}
