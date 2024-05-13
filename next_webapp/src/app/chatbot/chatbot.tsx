"use client";
import React, { useState } from "react";
import { IoChatbubbleEllipsesSharp, IoSend } from "react-icons/io5"; // Import IoSend for the send button
import { useRouter } from "next/navigation";
import "../chatbot/chatbot.css";


type Message = {
  content: React.ReactNode;  // Using React.ReactNode to accept both strings and JSX
  sender: string;
};

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [userInput, setUserInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([  
    {
      content: (
        <>
          Hi, How can I help you? Check out our <a href="/en/faq" style={{ color: 'blue', textDecoration: 'underline' }}>FAQ page</a> for more information.
        </>
      ),
      sender: "bot"
    },
  ]);
  const router = useRouter();

  const toggleChat = () => setIsOpen(!isOpen);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setUserInput(event.target.value);
  };

  const handleSend = () => {
    if (!userInput.trim()) return;
    const trimmedInput = userInput.trim().toLowerCase();
    setMessages([...messages, { content: userInput, sender: "user" }]);
    handleCommand(trimmedInput);
    setUserInput("");
  };

  const handleCommand = (input: string) => {
    const keywords = [
      {
        key: [
          "usecase",
          "usecases",
          "show me use cases",
          "use case page",
          "use cases",
        ],
        route: "/en/UseCases",
      },
      {
        key: ["about us", "aboutus", "about us page", "aboutus page"],
        route: "/en/about",
      },
      { key: ["statistics", "statistics page"], route: "/en/statistics" },
      { key: ["upload", "upload page", "uploadpage"], route: "/en/upload" },
      {
        key: ["sign up", "sign up page", "signup", "signup page"],
        route: "/en/signup",
      },
      { key: ["login", "login page"], route: "/en/login" },
      {
        key: [
          "resource-center",
          "resource-center page",
          "resourcecenter",
          "resource center page",
          "resource center",
        ],
        route: "/en/resource-center",
      },
      {
        key: ["datasets", "datasets page", "data sets page", "data sets"],
        route: "/en/datasets",
      },
      {
        key: [
          "contact",
          "contact page",
          "contact us page",
          "contact us",
          "contact us page",
        ],
        route: "/en/contact",
      },
      {
        key: [
          "privacypolicy",
          "privacypolicy page",
          "privacy policy",
          "privacy policy page",
        ],
        route: "/en/privacypolicy",
      },
      { key: ["licensing", "licensing page"], route: "/en/licensing" },
    ];

    const matchedKeyword = keywords.find(({ key }) =>
      key.some((keyword) => input.includes(keyword))
    );
    if (matchedKeyword) {
      setMessages([
        ...messages,
        { content: `Understood. Redirecting to the right page.`, sender: "bot" },
      ]);
      setTimeout(() => router.push(matchedKeyword.route), 2000);
    } else {
      setMessages([
        ...messages,
        { content: "Sorry, I didn't understand that.", sender: "bot" },
      ]);
    }
  };

  return (
    <div className="chatbot fixed bottom-4 right-4 flex flex-col items-end">
      {isOpen && (
        <div className="chat-window p-4 bg-white shadow-lg rounded-lg max-w-xs w-full">
          <div className="messages overflow-auto h-40">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.sender === "bot" ? "text-green-600" : "text-green-800"}`}>
                {/* Directly use msg.content without checking type */}
                {msg.content}  
              </div>
            ))}
          </div>
          <div className="input-area flex items-center mt-2">
            <input
              type="text"
              className="textarea1 flex-1 p-2 border rounded"
              onChange={handleInputChange}
              value={userInput}
              placeholder="Type a message..."
            />
            <button onClick={handleSend} className="send-icon ml-2">
              <IoSend className="text-green-500" size={24} />
            </button>
          </div>
        </div>
      )}
      <button
        onClick={toggleChat}
        className="toggle-btn text-3xl text-white bg-green-600 rounded-full p-3 hover:bg-green-700"
      >
        <IoChatbubbleEllipsesSharp />
      </button>
    </div>
  );
};

export default Chatbot;
