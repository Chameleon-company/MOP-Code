"use client";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/upload.css";
import { useEffect, useState, useRef } from "react";
import axios from "axios";
import { TagsInput } from "react-tag-input-component";

const Upload = () => {
  const inputRef = useRef<HTMLInputElement>(null);

  // Form state management
  const [isMounted, setIsMounted] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState("select");
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    trimester: "",
    tags: [],
  });
  const [validationErrors, setValidationErrors] = useState({
    name: "",
    description: "",
    trimester: "",
    file: "",
    tags: "", // Validation for tags
  });

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) {
    return null;
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setValidationErrors((prev) => ({ ...prev, file: "" }));
    if (event.target.files && event.target.files.length > 0) {
      const file = event.target.files[0];
      setSelectedFile(file);

      // Simulate file upload progress (you can use this interval for progress updates)
      const interval = setInterval(() => {
        setProgress((prev) => {
          const newProgress = prev + 5;
          if (newProgress >= 100) {
            clearInterval(interval);
            return 100;
          }
          return newProgress;
        });
      }, 500); // Update progress every 500 ms
    }
  };

  const onChooseFile = () => {
    inputRef.current?.click();
  };

  const clearFileInput = () => {
    if (inputRef.current) {
      inputRef.current.value = "";
    }
    setSelectedFile(null);
    setProgress(0);
    setUploadStatus("select");
  };

  const handleFormChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setValidationErrors((prev) => ({ ...prev, [name]: "" }));
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleTagsChange = (tags: string[]) => {
    // Clear the error for tags when there are tags
    setValidationErrors((prev) => ({
      ...prev,
      tags: tags.length > 0 ? "" : prev.tags,
    }));
    setFormData((prev) => ({
      ...prev,
      tags,
    }));
  };

  const handleUploadUseCase = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const errors: any = {};

    if (!formData.name) errors.name = "Name is required.";
    if (!formData.description) errors.description = "Description is required.";
    if (!formData.trimester) errors.trimester = "Trimester is required.";
    if (!selectedFile) {
      errors.file = "File is required.";
    } else if (
      selectedFile.type !== "text/x-python-script" &&
      !selectedFile.name.endsWith(".py")
    ) {
      errors.file = "Only Python files (.py) are allowed.";
    }
    if (formData.tags.length === 0) {
      errors.tags = "At least one tag is required.";
    }

    setValidationErrors(errors);

    if (Object.keys(errors).length > 0) {
      return;
    }

    try {
      // Create a file path (example path to public/uploads)
      const filePath = `uploads/${selectedFile.name}`;

      // Prepare form data
      const uploadFormData = new FormData();
      uploadFormData.append("file", selectedFile);
      uploadFormData.append("name", formData.name);
      uploadFormData.append("description", formData.description);
      uploadFormData.append("trimester", formData.trimester);
      uploadFormData.append("tags", JSON.stringify(formData.tags));
      uploadFormData.append("filePath", filePath); // Include file path in form data

      const response = await fetch("/api/upload", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: formData.name,
          description: formData.description, 
          trimester: formData.trimester, 
          tags: formData.tags, 
          filePath: filePath, 
        }),
      });

      if (response.status === 201) {
        alert("File uploaded successfully!");
        clearFileInput();
        setFormData({
          name: "",
          description: "",
          trimester: "",
          tags: [],
        });
      }
    } catch (error) {
      const errorMessage = "Error uploading file. Please try again.";
      alert(errorMessage);
      console.error("Upload error:", error);
    }
  };
  return (
    <div className="bg-gray-200">
      <Header />

      <div className="bg-gray-200 flex justify">
        <div className="upload-header-left">
          <h1 className="font-bold text-[50px] py-11">Upload Use Cases</h1>
        </div>
      </div>

      <div className="upload-container">
        <form onSubmit={handleUploadUseCase}>
          <div className="form-container">
            <div className="column">
              <label htmlFor="name">Name</label>
              <input
                type="text"
                id="name"
                name="name"
                value={formData.name}
                onChange={handleFormChange}
                placeholder="Enter name"
              />
              {validationErrors.name && (
                <p className="text-red-600 text-sm">{validationErrors.name}</p>
              )}

              <label htmlFor="tags">Tags</label>
              <TagsInput
                value={formData.tags}
                onChange={handleTagsChange}
                name="tags"
                placeHolder="Add tags and press enter"
                addOnPaste
              />
              {validationErrors.tags && (
                <p className="text-red-600 text-sm">{validationErrors.tags}</p>
              )}
            </div>

            <div className="column">
              <label htmlFor="description" className="ml-5">
                Description
              </label>
              <input
                type="text"
                id="description"
                name="description"
                className="ml-5"
                value={formData.description}
                onChange={handleFormChange}
                placeholder="Enter description"
              />
              {validationErrors.description && (
                <p className="text-red-600 text-sm">
                  {validationErrors.description}
                </p>
              )}

              <div className="column m-0">
                <label
                  htmlFor="trimester"
                  className=" text-lg font-medium text-gray-700"
                >
                  Trimester
                </label>
              </div>
              <div className="column m-0">
                <select
                  className="border border-gray-300 rounded-md px-[82%] py-3  my-1"
                  name="trimester"
                  id="trimester"
                  value={formData.trimester}
                  onChange={handleFormChange}
                >
                  <option value="">Select Trimester</option>
                  <option value="Trimester 1">Trimester 1</option>
                  <option value="Trimester 2">Trimester 2</option>
                  <option value="Trimester 3">Trimester 3</option>
                </select>
              </div>
              {validationErrors.trimester && (
                <p className="text-red-600 text-sm">
                  {validationErrors.trimester}
                </p>
              )}
            </div>
          </div>

          <div className="flex justify-center">
            <div className="border-dashed border-2 border-black w-[50rem] h-[25rem] items-center">
              <div className="flex items-center justify-center pt-[8rem]">
                <input
                  ref={inputRef}
                  type="file"
                  onChange={handleFileChange}
                  style={{ display: "none" }}
                />
                <button type="button" onClick={onChooseFile}>
                  <img
                    className="h-20 w-auto"
                    src="../img/Upload_use_case.png"
                    alt="Logo"
                  />
                </button>
              </div>
              <h1 className="text-center text-lg py-[2rem]">
                Click on logo to upload files
              </h1>
            </div>
            {validationErrors.file && (
              <p className="text-red-600 text-sm text-center pt-4">
                {validationErrors.file}
              </p>
            )}
          </div>

          {selectedFile && (
            <div className="flex pt-8 justify-center items-center">
              <div className="bg-gray-200 w-[50rem] py-[3rem] px-2 rounded-3xl relative">
                <div className="flex">
                  <h6 className="flex-1 font-bold items-center pl-2">
                    {selectedFile.name}
                  </h6>
                  <p>{progress}%</p>
                  <button
                    type="button"
                    className="text-red-600 absolute top-2 right-5"
                    onClick={clearFileInput}
                  >
                    Remove
                  </button>
                </div>
                <div className="w-full pt-3 bg-gray-200 rounded-full h-2.5 dark:bg-gray-700 pl-2">
                  <div
                    className="bg-green-600 h-2.5 rounded-full"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
              </div>
            </div>
          )}

          <div className="flex justify-center pt-4">
            <button
              type="submit"
              className="bg-green-500 text-white px-4 py-2 rounded"
            >
              {uploadStatus === "done" ? "Clear" : "Submit"}
            </button>
          </div>
        </form>
      </div>

      <Footer />
    </div>
  );
};

export default Upload;
