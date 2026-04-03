function sendMessage() {
    var userInput = document.getElementById("userInput").value;
    if (userInput === "") return;
    addMessage(userInput, "user-message");
    fetch("/send_message", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        body: "message=" + encodeURIComponent(userInput)
    })
    .then(response => response.json())
    .then(data => {
        if (data && data.length > 0) {
            addMessage(data[0].text, "bot-message");
        }
    })
    .catch(error => console.error("Error:", error));
    document.getElementById("userInput").value = "";
}

function addMessage(message, className) {
    var chatbox = document.getElementById("chatbox");
    var messageDiv = document.createElement("div");
    messageDiv.className = className;
    messageDiv.textContent = message;
    chatbox.appendChild(messageDiv);
    chatbox.scrollTop = chatbox.scrollHeight;
}
