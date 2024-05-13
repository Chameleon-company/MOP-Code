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
                <h1> Upload Case Studies</h1>
            </div>
            <div className="upload-header-right">
                <select style={{ border: "none" }}>
                                <option value="option1">Trimester 1</option>
                                <option value="option2">Trimester 2</option>
                                <option value="option3">Trimester 3</option>
                            </select>
            </div>
        </div>
        <div className="upload-container">

            <h2 style={{ textAlign: "left" }}>Uploader's Details</h2>
            <div className="form-container">
                <div className="column">
                    <label htmlFor="first-name">Author's Name</label>
                    <input type="text" id="first-name" name="first-name" placeholder="Enter your first name" />

                    <label htmlFor="last-name">DOP</label>
                    <input type="text" id="last-name" name="last-name" placeholder="Enter your last name" />

                    <label htmlFor="email">Company Email</label>
                    <input type="email" id="email" name="email" placeholder="Enter your email" />
                </div>

                <div className="column">
                    <label htmlFor="phone">Case Study</label>
                    <input type="tel" id="phone" name="phone" placeholder="Enter your phone number" />

                    <label htmlFor="address">Category</label>
                    <input type="text" id="address" name="address" placeholder="Enter your address" />

                </div>
            </div>
            <div className="green-box">
                <svg className="upload-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0-14c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6zm3 9h-6v-2h6v2z" />
                        </svg>
                <div className="file-upload">
                    <h1>Drag & Drop Files</h1>
                    <input type="file" id="file-upload" className="file-upload-input" />
                    <button className="upload-btn">UPLOAD FILES</button>
                </div>
            </div>

        </div>



    </main>
    <Footer />
</div>
); }; export default Upload;