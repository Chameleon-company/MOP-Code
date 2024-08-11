document.getElementById('send-button').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    console.log('User input:', userInput);  // Debug log
    if (userInput.trim() === '') return;

    addMessage('You', userInput);
    document.getElementById('user-input').value = '';

    console.log('Sending message:', userInput);  // Debug log

    fetch('http://localhost:5005/webhooks/rest/webhook', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => {
        console.log('Received response:', response);  // Debug log
        return response.json();
    })
    .then(data => {
        console.log('Response data:', data);  // Debug log
        data.forEach(message => {
            if (message.text) {
                addMessage('Bot', message.text);
            }
            if (message.image) {
                addImage('Bot', message.image);
            }
        });
    })
    .catch(error => console.error('Error:', error));
}

function addMessage(sender, message) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function addImage(sender, imageUrl) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    messageElement.innerHTML = `<strong>${sender}:</strong> <img src="${imageUrl}" alt="Image" style="max-width: 100%;">`;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}
