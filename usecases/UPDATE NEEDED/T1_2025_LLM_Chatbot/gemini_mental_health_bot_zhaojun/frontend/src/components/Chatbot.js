import React, { useEffect, useRef, useState } from "react";
import "./Chatbot.css";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef(null);

  useEffect(() => {
    fetch("http://localhost:5001/start_chat")
      .then((res) => res.json())
      .then((data) => {
        setMessages([{ sender: "bot", text: data.response }]);
      });
  }, []);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const newMessages = [...messages, { sender: "user", text: input }];
    setMessages(newMessages);
    setIsLoading(true);
    setInput("");

    try {
      const res = await fetch("http://localhost:5001/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });

      const data = await res.json();
      setMessages([...newMessages, { sender: "bot", text: data.response }]);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages([
        ...newMessages,
        {
          sender: "bot",
          text: "Sorry, something went wrong. Please try again.",
        },
      ]);
    }

    setIsLoading(false);
  };

  return (
    <div className="chatbot-container">
      <div className="chat-header">Mental Health Chatbot</div>

      <div className="chat-messages" ref={chatContainerRef}>
        {messages.map((msg, index) => (
          <div key={index} className={`message-wrapper ${msg.sender}`}>
            <img
              src={
                msg.sender === "bot" ? "/bot-avatar.png" : "/user-avatar.png"
              }
              alt={`${msg.sender} avatar`}
              className="avatar"
            />
            <div className={`message ${msg.sender}`}>{msg.text}</div>
          </div>
        ))}
      </div>

      <div className="chat-input">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          rows={2}
        />
        <button onClick={sendMessage} disabled={isLoading}>
          {isLoading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
};

export default Chatbot;
