"use client";
import React, { useState } from "react";
import { IoChatbubbleEllipsesSharp, IoSend } from "react-icons/io5";
import { useRouter } from "next/navigation";
import "../chatbot/chatbot.css";
import nlp from 'compromise'; // Importing compromise for NLP

type Message = {
  content: React.ReactNode;
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
      sender: "bot",
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

  // Updated handleCommand function to use NLP
  const handleCommand = (input: string) => {
    const doc = nlp(input); // Creating an NLP document with the user input
    const keywords = [
      {
        key: ["use case", "usecases", "use case page", "use cases"],
        route: "/en/UseCases",
      },
      { key: ["about us", "aboutus", "about us page"], route: "/en/about" },
      { key: ["statistics", "statistics page"], route: "/en/statistics" },
      { key: ["upload", "upload page"], route: "/en/upload" },
      { key: ["sign up", "signup", "signup page"], route: "/en/signup" },
      { key: ["login", "login page"], route: "/en/login" },
      { key: ["resource center", "resourcecenter"], route: "/en/resource-center" },
      { key: ["datasets", "datasets page"], route: "/en/datasets" },
      { key: ["contact", "contact us"], route: "/en/contact" },
      { key: ["privacy policy", "privacypolicy"], route: "/en/privacypolicy" },
      { key: ["licensing", "licensing page"], route: "/en/licensing" },
    ];

    let foundMatch = false;

    // Using NLP to check if the input matches any of the keyword intents
    keywords.forEach(({ key, route }) => {
      key.forEach(keyword => {
        if (doc.has(keyword)) { // Using NLP's "has" method to find matches
          setMessages([
            ...messages,
            { content: `Understood. Redirecting to the right page.`, sender: "bot" },
          ]);
          setTimeout(() => router.push(route), 2000);
          foundMatch = true;
        }
      });
    });

    // If no matches found, it will then provide a fallback response
    if (!foundMatch) {
      setMessages([
        ...messages,
        { content: "Sorry, I didn't understand that. Can you try rephrasing?", sender: "bot" },
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

