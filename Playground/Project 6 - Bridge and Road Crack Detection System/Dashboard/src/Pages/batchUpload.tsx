import TheDrawer from '../Components/drawer'
import MultipleFileUploader from '../Components/multipleUpload'







export function BatchUpload() {

    return (
        <>
            <div><TheDrawer></TheDrawer></div>
            <h1>Upload a file for report generation</h1>
            <div>File MUST be a binary mask in either PNG or JPG/JPEG format!</div>
            <div>This page should be updated to allow for normal (non mask image) to be uploaded and mask should be auto generated! </div>
            <div><MultipleFileUploader></MultipleFileUploader></div>
            
        </>


    )
}
