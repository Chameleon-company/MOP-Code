// Add an event listener to the form with the ID 'upload-form'
document.getElementById('upload-form').addEventListener('submit', function(event) {
    // Prevent the default form submission behavior, which would reload the page
    event.preventDefault();

    // Get the file input element from the HTML
    const fileInput = document.getElementById('file-input');

    // Create a new FormData object to hold the file data
    const formData = new FormData();

    // Append the selected file to the FormData object
    // fileInput.files[0] accesses the first (and usually only) file selected by the user
    formData.append('file', fileInput.files[0]);

    // Use the Fetch API to send the file to the server
    fetch('/upload', {
        method: 'POST', // Specify that we are sending data to the server using the POST method
        body: formData // Include the FormData object as the body of the request
    })
    .then(response => response.blob()) // Convert the server's response to a Blob (binary large object)
    .then(blob => {
        // Create a URL for the Blob object that can be used to display it in the browser
        const url = URL.createObjectURL(blob);

        // Set the 'src' attribute of the 'output-image' element to the Blob URL
        // This will display the result image received from the server
        document.getElementById('output-image').src = url;

        // Make sure the 'output-image' element is visible
        document.getElementById('output-image').style.display = 'block';

        // Set the 'src' attribute of the 'input-image' element to a URL representing the selected file
        // This will display the original image selected by the user
        document.getElementById('input-image').src = URL.createObjectURL(fileInput.files[0]);

        // Make sure the 'input-image' element is visible
        document.getElementById('input-image').style.display = 'block';
    })
    .catch(error => console.error('Error:', error)); // Log any errors that occur during the fetch process
});
