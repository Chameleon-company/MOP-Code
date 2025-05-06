"use client";

import React, { useState } from "react";
import { IoChatbubbleEllipsesSharp, IoSend } from "react-icons/io5";
import { useRouter } from "next/navigation";
import "./chatbot.css";
import enMessages from "./en.json";
import { processInput } from "./nlp/nlpProcessor";

type Message = {
  content: React.ReactNode;
  sender: string;
};

// Define a type for a use case entry.
interface UseCase {
  title: string;
  description: string;
  tags: string[];
}

//Escapes special regex characters in a string.

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/*
 - Perform a local use-case search using whole-word regex.
 - Returns matching use cases if a whole-word match is found.
 */
const searchLocalUseCases = (query: string): UseCase[] => {
  const results: UseCase[] = [];
  const lowerQuery = query.toLowerCase().trim();
  if (lowerQuery.length < 3) return results; // skip if too short

  const details: { [key: string]: UseCase } = enMessages.use_case_details;
  // Build a regex to match the query as a whole word.
  const queryRegex = new RegExp(`\\b${escapeRegExp(lowerQuery)}\\b`, "i");

  Object.keys(details).forEach((key) => {
    const useCase = details[key];
    const titleMatch = queryRegex.test(useCase.title.toLowerCase());
    const descMatch = queryRegex.test(useCase.description.toLowerCase());
    const tagMatch = useCase.tags.some((tag: string) =>
      new RegExp(`\\b${escapeRegExp(tag.toLowerCase())}\\b`).test(lowerQuery)
    );

    if (titleMatch || descMatch || tagMatch) {
      results.push(useCase);
    }
  });
  return results;
};

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [userInput, setUserInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      content: (
        <>
          {enMessages.initial.welcome}
          <a
            href={enMessages.initial.faq_url}
            style={{ color: "blue", textDecoration: "underline" }}
          >
            {enMessages.initial.faq_link_text}
          </a>
          {enMessages.initial.more_info}
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

  //API-based use-case search.
  const fetchUseCasesFromAPI = async (searchTerm: string, searchMode: string = "TITLE") => {
    try {
      const response = await fetch("/api/search-use-cases", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category: "", searchMode, searchTerm }),
      });
      const data = await response.json();
      return data.filteredStudies;
    } catch (error) {
      console.error("Error fetching use cases: ", error);
      return [];
    }
  };

  const handleCommand = async (input: string) => {
    // Determine intent via regex-based NLP.
    const matchedIntent = processInput(input);

    switch (matchedIntent) {
      case "greet":
        setMessages((prev) => [
          ...prev,
          { content: <>{enMessages.greet.response}</>, sender: "bot" },
        ]);
        break;

      case "greet_morning":
        setMessages((prev) => [
          ...prev,
          { content: <>{enMessages.greet_morning.response}</>, sender: "bot" },
        ]);
        break;

      case "greet_afternoon":
        setMessages((prev) => [
          ...prev,
          { content: <>{enMessages.greet_afternoon.response}</>, sender: "bot" },
        ]);
        break;

      case "greet_evening":
        setMessages((prev) => [
          ...prev,
          { content: <>{enMessages.greet_evening.response}</>, sender: "bot" },
        ]);
        break;

      case "about_mop":
        setMessages((prev) => [
          ...prev,
          { content: <>{enMessages.about.p1}</>, sender: "bot" },
          { content: <>{enMessages.about.p2}</>, sender: "bot" },
        ]);
        break;

      case "mop_full_form":
        // The user specifically asked about MOP's full form & explanation
        setMessages((prev) => [
          ...prev,
          { content: <>{enMessages.mop_full_form.response}</>, sender: "bot" },
        ]);
        break;

      case "project_overview":
        setMessages((prev) => [
          ...prev,
          { content: <>{enMessages.project_overview.response}</>, sender: "bot" },
        ]);
        break;

      case "contact":
        setMessages((prev) => [
          ...prev,
          { content: <>{enMessages.contact.response}</>, sender: "bot" },
        ]);
        break;

      case "help":
        setMessages((prev) => [
          ...prev,
          { content: <>{enMessages.help.response}</>, sender: "bot" },
        ]);
        break;

      case "use_cases": {
        // Perform a local search for relevant use cases.
        const localResults = searchLocalUseCases(input);
        if (localResults.length > 0) {
          setMessages((prev) => [
            ...prev,
            {
              content: (
                <>
                  <div>{enMessages.use_cases.intro}</div>
                  <ul>
                    {localResults.map((cs, index) => (
                      <li key={index}>
                        <strong>{cs.title}</strong>: {cs.description}
                      </li>
                    ))}
                  </ul>
                </>
              ),
              sender: "bot",
            },
          ]);
        } else {
          // If no local results, try API or prompt for more details
          const apiResults = await fetchUseCasesFromAPI(input);
          if (apiResults.length > 0) {
            setMessages((prev) => [
              ...prev,
              {
                content: (
                  <>
                    <div>{enMessages.use_cases.intro}</div>
                    <ul>
                      {apiResults.map((cs: any) => (
                        <li key={cs.id}>
                          <strong>{cs.name}</strong>: {cs.description}
                        </li>
                      ))}
                    </ul>
                  </>
                ),
                sender: "bot",
              },
            ]);
          } else {
            setMessages((prev) => [
              ...prev,
              { content: <>{enMessages.use_case_prompt.response}</>, sender: "bot" },
            ]);
          }
        }
        break;
      }

      case "faq":
        window.location.href = enMessages.initial.faq_url;
        break;

      // Navigation commands
      case "navigate_home":
        router.push("/en/");
        break;

      case "navigate_about":
        router.push("/en/about");
        break;

      case "navigate_contact":
        router.push("/en/contact");
        break;

      case "navigate_statistics":
        router.push("/en/statistics");
        break;

      case "navigate_upload":
        router.push("/en/upload");
        break;

      case "navigate_language":
        // Show available language suggestions
        setMessages((prev) => [
          ...prev,
          {
            content: (
              <>
                <div>{enMessages.language_prompt.response}</div>
                <ul>
                  <li>English</li>
                  <li>中文</li>
                  <li>Español</li>
                  <li>Ελληνικά</li>
                </ul>
              </>
            ),
            sender: "bot",
          },
        ]);
        break;

      case "navigate_sign_up":
        // Redirect to /en/signup
        router.push("/en/signup");
        break;

      case "navigate_log_in":
        // Redirect to /en/login
        router.push("/en/login");
        break;

      case "thank_you":
        setMessages((prev) => [
          ...prev,
          { content: <>{enMessages.thank_you.response}</>, sender: "bot" },
        ]);
        break;

      case "goodbye":
      case "bye":
        setMessages((prev) => [
          ...prev,
          { content: <>{enMessages.bye.response}</>, sender: "bot" },
        ]);
        break;

      default: {
        // Fallback: If no intent matched, try API search or show fallback message
        const results = await fetchUseCasesFromAPI(input);
        if (results.length > 0) {
          setMessages((prev) => [
            ...prev,
            {
              content: (
                <>
                  <div>{enMessages.use_cases.intro}</div>
                  <ul>
                    {results.map((cs: any) => (
                      <li key={cs.id}>
                        <strong>{cs.name}</strong>: {cs.description}
                      </li>
                    ))}
                  </ul>
                </>
              ),
              sender: "bot",
            },
          ]);
        } else {
          setMessages((prev) => [
            ...prev,
            { content: <>{enMessages.fallback.response}</>, sender: "bot" },
          ]);
        }
        break;
      }
    }
  };

  const handleSend = async () => {
    if (!userInput.trim()) {
      setMessages((prev) => [
        ...prev,
        { content: <>{enMessages.validation.empty_input}</>, sender: "bot" },
      ]);
      return;
    }
    const trimmedInput = userInput.trim();
    setMessages((prev) => [
      ...prev,
      { content: <>{userInput}</>, sender: "user" },
    ]);
    await handleCommand(trimmedInput);
    setUserInput("");
  };

  return (
    <div className="chatbot fixed bottom-4 right-4 flex flex-col items-end">
      {isOpen && (
        <div className="chat-window p-4 bg-white shadow-lg rounded-lg max-w-xs w-full">
          <div className="messages overflow-auto h-40">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`message ${
                  msg.sender === "bot" ? "text-green-600" : "text-green-800"
                }`}
              >
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
              aria-label="User input"
            />
            <button onClick={handleSend} className="send-icon ml-2" aria-label="Send message">
              <IoSend className="text-green-500" size={24} />
            </button>
          </div>
        </div>
      )}
      <button
        onClick={toggleChat}
        className="toggle-btn text-3xl text-white bg-green-600 rounded-full p-3 hover:bg-green-700"
        aria-label="Open chat"
      >
        <IoChatbubbleEllipsesSharp />
      </button>
    </div>
  );
};

export default Chatbot;
