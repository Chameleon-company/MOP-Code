import TheDrawer from '../Components/drawer'
import FileUploader from '../Components/singleUpload'







export function Uploader() {

    return (
        <>
            <div><TheDrawer></TheDrawer></div>
            <h1>Upload a file for report generation</h1>
            <div>File MUST be a binary mask in either PNG or JPG/JPEG format!</div>
            <div>This page should be updated to allow for normal (non mask image) to be uploaded and mask should be auto generated! </div>
            <div><FileUploader></FileUploader></div>
            
        </>


    )
}
