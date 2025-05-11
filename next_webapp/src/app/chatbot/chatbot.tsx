"use client";

import React, { useState, useEffect, useRef } from "react";
import { IoChatbubbleEllipsesSharp, IoSend } from "react-icons/io5";
import {
  BsMicFill,
  BsMicMuteFill,
  BsVolumeUp,
  BsVolumeMute,
} from "react-icons/bs";
import { useRouter } from "next/navigation";
import "./chatbot.css";
import enMessages from "./en.json";
import { processInput } from "./nlp/nlpProcessor";

type Message = {
  content: React.ReactNode;
  sender: string;
  text?: string;
};

// Define a type for a use case entry.
interface UseCase {
  title: string;
  description: string;
  tags: string[];
}

//Escapes special regex characters in a string.
function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
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
      text:
        enMessages.initial.welcome +
        enMessages.initial.faq_link_text +
        enMessages.initial.more_info,
    },
  ]);

  const router = useRouter();
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [recognitionSupported, setRecognitionSupported] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);
  const recognitionRef = useRef<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const speechSynthRef = useRef<SpeechSynthesis | null>(null);

  // Check for browser speech support
  useEffect(() => {
    // Check for SpeechRecognition support
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      setRecognitionSupported(true);
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = "en-US";

      recognitionRef.current.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setUserInput(transcript);
        // Auto-send the recognized speech
        setTimeout(() => {
          handleSendVoiceInput(transcript);
        }, 500);
      };

      recognitionRef.current.onerror = (event: any) => {
        console.error("Speech recognition error:", event.error);
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }

    // Check for Speech Synthesis support
    if (typeof window !== "undefined" && window.speechSynthesis) {
      setSpeechSupported(true);
      speechSynthRef.current = window.speechSynthesis;
    }

    return () => {
      // Cleanup
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
      if (speechSynthRef.current) {
        speechSynthRef.current.cancel();
      }
    };
  }, []);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const toggleChat = () => setIsOpen(!isOpen);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setUserInput(event.target.value);
  };

  const toggleListening = () => {
    if (isListening) {
      recognitionRef.current?.abort();
      setIsListening(false);
    } else {
      try {
        recognitionRef.current?.start();
        setIsListening(true);
      } catch (error) {
        console.error("Error starting speech recognition:", error);
      }
    }
  };

  const speakMessage = (text: string) => {
    if (!speechSynthRef.current) return;

    // Cancel any ongoing speech
    speechSynthRef.current.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = (event) => {
      console.error("Speech synthesis error:", event);
      setIsSpeaking(false);
    };

    speechSynthRef.current.speak(utterance);
  };

  const stopSpeaking = () => {
    if (speechSynthRef.current) {
      speechSynthRef.current.cancel();
      setIsSpeaking(false);
    }
  };

  //use-case search.
  const fetchUseCasesFromAPI = async (
    searchTerm: string,
    searchMode: string = "TITLE"
  ) => {
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

  // Extract plain text from JSX for text-to-speech
  const extractTextFromJSX = (jsxContent: React.ReactNode): string => {
    if (typeof jsxContent === "string") {
      return jsxContent;
    } else if (Array.isArray(jsxContent)) {
      return jsxContent.map((item) => extractTextFromJSX(item)).join(" ");
    } else if (React.isValidElement(jsxContent)) {
      const { children } = jsxContent.props;
      return extractTextFromJSX(children);
    } else if (jsxContent === null || jsxContent === undefined) {
      return "";
    } else if (typeof jsxContent === "object") {
      return Object.values(jsxContent)
        .map((item) => extractTextFromJSX(item))
        .join(" ");
    }
    return String(jsxContent);
  };

  const handleCommand = async (input: string) => {
    // Determine intent via regex-based NLP.
    const matchedIntent = processInput(input);
    let responseText = "";

    switch (matchedIntent) {
      case "greet":
        responseText = enMessages.greet.response;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      case "greet_morning":
        responseText = enMessages.greet_morning.response;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      case "greet_afternoon":
        responseText = enMessages.greet_afternoon.response;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      case "greet_evening":
        responseText = enMessages.greet_evening.response;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      case "about_mop":
        responseText = enMessages.about.p1 + " " + enMessages.about.p2;
        setMessages((prev) => [
          ...prev,
          {
            content: (
              <>
                <p>{enMessages.about.p1}</p>
                <p>{enMessages.about.p2}</p>
              </>
            ),
            sender: "bot",
            text: responseText,
          },
        ]);
        break;

      case "mop_full_form":
        responseText = enMessages.mop_full_form.response;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      case "project_overview":
        responseText = enMessages.project_overview.response;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      case "contact":
        responseText = enMessages.contact.response;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      case "help":
        responseText = enMessages.help.response;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      case "use_cases": {
        // Perform a local search for relevant use cases.
        const localResults = searchLocalUseCases(input);
        if (localResults.length > 0) {
          responseText = `${enMessages.use_cases.intro} ${localResults
            .map((cs) => `${cs.title}: ${cs.description}`)
            .join(". ")}`;
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
              text: responseText,
            },
          ]);
        } else {
          // If no local results, try API or prompt for more details
          const apiResults = await fetchUseCasesFromAPI(input);
          if (apiResults.length > 0) {
            responseText = `${enMessages.use_cases.intro} ${apiResults
              .map((cs: any) => `${cs.name}: ${cs.description}`)
              .join(". ")}`;
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
                text: responseText,
              },
            ]);
          } else {
            responseText = enMessages.use_case_prompt.response;
            setMessages((prev) => [
              ...prev,
              {
                content: <>{responseText}</>,
                sender: "bot",
                text: responseText,
              },
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
        responseText = enMessages.navigation.home;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      case "navigate_about":
        router.push("/en/about");
        responseText = enMessages.navigation.about;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      case "navigate_contact":
        router.push("/en/contact");
        responseText = enMessages.navigation.contact;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      case "navigate_statistics":
        router.push("/en/statistics");
        responseText = enMessages.navigation.statistics;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      case "navigate_upload":
        router.push("/en/upload");
        responseText = enMessages.navigation.upload;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      case "navigate_language":
        // Show available language suggestions
        responseText = `${
          enMessages.language_prompt.response
        } ${enMessages.language_prompt.languages.join(", ")}`;
        setMessages((prev) => [
          ...prev,
          {
            content: (
              <>
                <div>{enMessages.language_prompt.response}</div>
                <ul>
                  {enMessages.language_prompt.languages.map((lang, index) => (
                    <li key={index}>{lang}</li>
                  ))}
                </ul>
              </>
            ),
            sender: "bot",
            text: responseText,
          },
        ]);
        break;

      case "navigate_sign_up":
        // Redirect to /en/signup
        router.push("/en/signup");
        responseText = enMessages.navigation.sign_up;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      case "navigate_log_in":
        // Redirect to /en/login
        router.push("/en/login");
        responseText = enMessages.navigation.log_in;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      case "thank_you":
        responseText = enMessages.thank_you.response;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      case "goodbye":
      case "bye":
        responseText = enMessages.bye.response;
        setMessages((prev) => [
          ...prev,
          { content: <>{responseText}</>, sender: "bot", text: responseText },
        ]);
        break;

      default: {
        // Fallback: If no intent matched, try API search or show fallback message
        const results = await fetchUseCasesFromAPI(input);
        if (results.length > 0) {
          responseText = `${enMessages.use_cases.intro} ${results
            .map((cs: any) => `${cs.name}: ${cs.description}`)
            .join(". ")}`;
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
              text: responseText,
            },
          ]);
        } else {
          responseText = enMessages.fallback.response;
          setMessages((prev) => [
            ...prev,
            { content: <>{responseText}</>, sender: "bot", text: responseText },
          ]);
        }
        break;
      }
    }

    // Speak the response if TTS is enabled
    if (responseText && isSpeaking) {
      speakMessage(responseText);
    }

    return responseText;
  };

  const handleSend = async () => {
    if (!userInput.trim()) {
      const emptyInputMessage = enMessages.validation.empty_input;
      setMessages((prev) => [
        ...prev,
        {
          content: <>{emptyInputMessage}</>,
          sender: "bot",
          text: emptyInputMessage,
        },
      ]);
      if (isSpeaking) {
        speakMessage(emptyInputMessage);
      }
      return;
    }
    const trimmedInput = userInput.trim();
    setMessages((prev) => [
      ...prev,
      { content: <>{userInput}</>, sender: "user", text: userInput },
    ]);
    setUserInput("");
    await handleCommand(trimmedInput);
  };

  const handleSendVoiceInput = async (transcript: string) => {
    if (!transcript.trim()) return;
    setMessages((prev) => [
      ...prev,
      { content: <>{transcript}</>, sender: "user", text: transcript },
    ]);
    await handleCommand(transcript);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSend();
    }
  };

  const toggleSpeech = () => {
    if (isSpeaking) {
      stopSpeaking();
    } else {
      setIsSpeaking(true);
      // Speak the last bot message if available
      const lastBotMessage = [...messages]
        .reverse()
        .find((msg) => msg.sender === "bot");
      if (lastBotMessage?.text) {
        speakMessage(lastBotMessage.text);
      }
    }
  };

  return (
    <div className="chatbot fixed bottom-4 right-4 flex flex-col items-end z-50">
      {isOpen && (
        <div className="chat-window p-4 bg-white shadow-lg rounded-lg max-w-xs w-full">
          <div className="flex justify-between items-center mb-2 border-b pb-2">
            <h3 className="text-md font-semibold text-green-600">
              Melbourne Open Data Assistant
            </h3>
            <div className="flex space-x-2">
              {speechSupported && (
                <button
                  onClick={toggleSpeech}
                  className={`p-1 rounded ${isSpeaking ? "bg-green-100" : ""}`}
                  aria-label={isSpeaking ? "Mute voice" : "Enable voice"}
                  title={isSpeaking ? "Mute voice" : "Enable voice"}
                >
                  {isSpeaking ? (
                    <BsVolumeUp className="text-green-600" />
                  ) : (
                    <BsVolumeMute className="text-gray-500" />
                  )}
                </button>
              )}
              {recognitionSupported && (
                <button
                  onClick={toggleListening}
                  className={`p-1 rounded ${
                    isListening ? "bg-green-100 animate-pulse" : ""
                  }`}
                  aria-label={
                    isListening ? "Stop listening" : "Start voice input"
                  }
                  title={isListening ? "Stop listening" : "Start voice input"}
                >
                  {isListening ? (
                    <BsMicFill className="text-green-600" />
                  ) : (
                    <BsMicMuteFill className="text-gray-500" />
                  )}
                </button>
              )}
            </div>
          </div>
          <div className="messages overflow-auto h-60">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`message p-2 my-1 rounded-lg ${
                  msg.sender === "bot"
                    ? "bg-green-50 text-green-800 border-l-4 border-green-500"
                    : "bg-gray-100 text-gray-800 text-right"
                }`}
              >
                {msg.content}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
          <div className="input-area flex items-center mt-3 border-t pt-3">
            <input
              type="text"
              className="flex-1 p-2 border rounded-l outline-none focus:ring-2 focus:ring-green-300"
              onChange={handleInputChange}
              value={userInput}
              placeholder={isListening ? "Listening..." : "Type a message..."}
              onKeyPress={handleKeyPress}
              disabled={isListening}
              aria-label="User input"
            />
            <button
              onClick={handleSend}
              className="send-icon bg-green-500 text-white p-2 rounded-r hover:bg-green-600 transition duration-150"
              aria-label="Send message"
            >
              <IoSend size={20} />
            </button>
          </div>
          {isListening && (
            <div className="text-xs text-center mt-1 text-green-600">
              Listening... Speak now!
            </div>
          )}
          <div className="text-xs text-center mt-2 text-gray-500">
            {recognitionSupported
              ? "Voice commands available. Click the mic icon to speak."
              : "Voice recognition not supported in your browser."}
          </div>
        </div>
      )}
      <button
        onClick={toggleChat}
        className="toggle-btn text-3xl text-white bg-green-600 rounded-full p-3 hover:bg-green-700 shadow-lg transition-transform hover:scale-105"
        aria-label="Open chat"
      >
        <IoChatbubbleEllipsesSharp />
      </button>
    </div>
  );
};

export default Chatbot;
