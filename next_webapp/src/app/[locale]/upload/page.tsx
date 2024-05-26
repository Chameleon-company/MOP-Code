import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import "../../../../public/styles/upload.css";
import { useTranslations } from "next-intl";

const Upload = () => {
  const t = useTranslations("upload");

  return (
    <div>
      <Header />
      <main className="upload-body">
        <div className="upload-header">
          <div className="upload-header-left">
            <h1>{"Upload Case Studies"}</h1>
          </div>
          <div className="upload-header-right">
            <select style={{ border: "none" }}>
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
          <div className="green-box">
            <svg width="800px" height="800px" viewBox="0 0 1024 1024" className="upload-icon" version="1.1" xmlns="http://www.w3.org/2000/svg">
              <path d="M736.68 435.86a173.773 173.773 0 0 1 172.042 172.038c0.578 44.907-18.093 87.822-48.461 119.698-32.761 34.387-76.991 51.744-123.581 52.343-68.202 0.876-68.284 106.718 0 105.841 152.654-1.964 275.918-125.229 277.883-277.883 1.964-152.664-128.188-275.956-277.883-277.879-68.284-0.878-68.202 104.965 0 105.842zM285.262 779.307A173.773 173.773 0 0 1 113.22 607.266c-0.577-44.909 18.09-87.823 48.461-119.705 32.759-34.386 76.988-51.737 123.58-52.337 68.2-0.877 68.284-106.721 0-105.842C132.605 331.344 9.341 454.607 7.379 607.266 5.417 759.929 135.565 883.225 285.262 885.148c68.284 0.876 68.2-104.965 0-105.841z" fill="#4A5699" />
              <path d="M339.68 384.204a173.762 173.762 0 0 1 172.037-172.038c44.908-0.577 87.822 18.092 119.698 48.462 34.388 32.759 51.743 76.985 52.343 123.576 0.877 68.199 106.72 68.284 105.843 0-1.964-152.653-125.231-275.917-277.884-277.879-152.664-1.962-275.954 128.182-277.878 277.879-0.88 68.284 104.964 68.199 105.841 0z" fill="#C45FA0" />
              <path d="M545.039 473.078c16.542 16.542 16.542 43.356 0 59.896l-122.89 122.895c-16.542 16.538-43.357 16.538-59.896 0-16.542-16.546-16.542-43.362 0-59.899l122.892-122.892c16.537-16.542 43.355-16.542 59.894 0z" fill="#F39A2B" />
              <path d="M485.17 473.078c16.537-16.539 43.354-16.539 59.892 0l122.896 122.896c16.538 16.533 16.538 43.354 0 59.896-16.541 16.538-43.361 16.538-59.898 0L485.17 532.979c-16.547-16.543-16.547-43.359 0-59.901z" fill="#F39A2B" />
              <path d="M514.045 634.097c23.972 0 43.402 19.433 43.402 43.399v178.086c0 23.968-19.432 43.398-43.402 43.398-23.964 0-43.396-19.432-43.396-43.398V677.496c0.001-23.968 19.433-43.399 43.396-43.399z" fill="#E5594F" />
            </svg>
            <div className="file-upload">
              <h1>{"Drag & Drop Files"}</h1>
              <input type="file" id="file-upload" className="file-upload-input" />
              <button className="upload-btn">{"UPLOAD FILES"}</button>
            </div>
          </div>
        </div>
        <div className="spacer">
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Upload;
