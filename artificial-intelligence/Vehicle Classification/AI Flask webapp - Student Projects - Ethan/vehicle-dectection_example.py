from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO  # Import YOLOv8 from ultralytics
from PIL import Image

app = Flask(__name__) # Create a Flask application instance
app.config['UPLOAD_FOLDER'] = 'uploads/' # Specifies the folder where images uploaded to the webapp will be stored
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'} # Specifies allowed file types for upload to the webapp

# Load YOLOv8 model
model = YOLO('/Users/brockalexiadis/Documents/Python/YOLOv8 Test and Train/runs/detect/train11/weights/last.pt')  # Update with the correct path to your YOLOv8 model

# Checks if the uploaded image is one of the allowed file types
def allowed_file(filename):
    # Check if there's a '.' in the filename and if the file extension is in the allowed set
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/') # Home page route
def index():
    return render_template('index.html', active_page="index") # Uses our index.html for the homepage

# Handles uploads to the webpage
@app.route('/upload', methods=['POST']) # Data from user is sent to webpage through POST method
def upload_file():
    # Check if 'file' part is in what the user uploads; if not, give error
    if 'file' not in request.files:
        print("Error: No file part in request.")  # Debug statement
        return jsonify({'error': 'No file part'})

    # Retrieve the file from the request
    file = request.files['file']

    # Check if file name is empty, if it is, return an error
    if file.filename == '':
        print("Error: No selected file.")  # Debug statement
        return jsonify({'error': 'No selected file'})

    # If it has a file name, check if it's allowed and then proceed to process the file
    if file and allowed_file(file.filename):
        # Secure filename to prevent attacks
        filename = secure_filename(file.filename)
        # Path to which the file will be saved
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Save the file to that path
        file.save(filepath)

        # Check what file type the user uploaded
        if filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}:
            # Open image using PIL (Python Imaging Library)
            image = Image.open(filepath)
            # Use our YOLO model to do vehicle detection
            results = model(image)

            # Check if results is a list and handle it
            if isinstance(results, list):
                # If results is a list, iterate through each result in it
                for i, result in enumerate(results):
                    # Create a file path for saving each result image. the filename includes an index to differentiate them
                    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'detected_{i}_{filename}')
                    # Save the result image with detections to the specified path
                    result.save(result_image_path)
            else:
                # If results is not a list (i.e., a single result), create a file path for the result image
                result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + filename)
                # Save the single result image with detections to the specified path
                results.save(result_image_path)

            # Send the saved result image back to the client, allowing it to be downloaded
            return send_from_directory(
                directory=app.config['UPLOAD_FOLDER'], # The directory where the result images are stored
                path=os.path.basename(result_image_path), # The basename of the result image file (just the filename)
                as_attachment=True) # Indicate that the file should be sent as an attachment (i.e., a download)
        else:
            # If the file type is unsupported, return an error message as JSON
            print("Error: Unsupported file type for processing.")  # Debug statement
            return jsonify({'error': 'Unsupported file type for processing'})

    # If no file was uploaded or the file type is invalid, return an error message as JSON
    print("Error: Invalid file type.")  # Debug statement
    return jsonify({'error': 'Invalid file type'})

# This block checks if the script is being run directly (not imported as a module)
if __name__ == '__main__':
    # Run app in debug mode
    app.run(host='0.0.0.0', port=5001, debug=False)

