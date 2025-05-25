"use client";
import React, { useState, useEffect, useRef } from "react";
import { IoChatbubbleEllipsesSharp, IoSend } from "react-icons/io5";
import { BsMicFill, BsMicMuteFill, BsVolumeUp, BsVolumeMute } from "react-icons/bs";
import { useRouter } from "next/navigation";
import enMessages from "./en.json";
import "./chatbot.css";
import { processInput } from "./nlp/nlpProcessor";
import { CaseStudy } from "@/app/types";

type Message = {
  content: React.ReactNode;
  sender: string;
  text?: string;
};

interface UseCase {
  title: string;
  description: string;
  tags: string[];
}

interface LiveUseCase {
  id: number;
  name: string;
  description: string;
  filename: string;
  tags: string[];
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const searchLocalUseCases = (query: string): UseCase[] => {
  const results: UseCase[] = [];
  const lowerQuery = query.toLowerCase().trim();
  if (lowerQuery.length < 3) return results;

  const details: { [key: string]: UseCase } = enMessages.use_case_details;
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

const extractTextFromJSX = (jsxContent: React.ReactNode): string => {
  if (typeof jsxContent === "string") return jsxContent;
  if (Array.isArray(jsxContent)) return jsxContent.map(extractTextFromJSX).join(" ");
  if (React.isValidElement(jsxContent)) return extractTextFromJSX(jsxContent.props.children);
  if (jsxContent === null || jsxContent === undefined) return "";
  if (typeof jsxContent === "object") return Object.values(jsxContent).map(extractTextFromJSX).join(" ");
  return String(jsxContent);
};

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [userInput, setUserInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      content: (
        <>
          {enMessages.initial.welcome}
          <a href={enMessages.initial.faq_url} style={{ color: "blue", textDecoration: "underline" }}>
            {enMessages.initial.faq_link_text}
          </a>
          {enMessages.initial.more_info}
        </>
      ),
      sender: "bot",
      text: enMessages.initial.welcome + enMessages.initial.faq_link_text + enMessages.initial.more_info,
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

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      setRecognitionSupported(true);
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = "en-US";

      recognitionRef.current.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setUserInput(transcript);
        setTimeout(() => handleSendVoiceInput(transcript), 500);
      };

      recognitionRef.current.onerror = () => setIsListening(false);
      recognitionRef.current.onend = () => setIsListening(false);
    }

    if (typeof window !== "undefined" && window.speechSynthesis) {
      setSpeechSupported(true);
      speechSynthRef.current = window.speechSynthesis;
    }

    return () => {
      recognitionRef.current?.abort();
      speechSynthRef.current?.cancel();
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const toggleChat = () => setIsOpen(!isOpen);
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => setUserInput(e.target.value);

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
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    speechSynthRef.current.speak(utterance);
  };

  const stopSpeaking = () => {
    speechSynthRef.current?.cancel();
    setIsSpeaking(false);
  };

  const fetchUseCasesFromAPI = async (searchTerm: string, searchMode = "TITLE") => {
    try {
      const response = await fetch("/api/search-use-cases", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category: "", searchMode, searchTerm }),
      });
      const data = await response.json();
      return data.filteredStudies;
    } catch (error) {
      console.error("Error fetching use cases:", error);
      return [];
    }
  };

  const handleCommand = async (input: string) => {
    const intent = processInput(input);
    let responseText = "";

    if (["greet", "greet_morning", "greet_afternoon", "greet_evening", "mop_full_form", "project_overview", "contact", "help", "thank_you", "bye", "goodbye"].includes(intent)) {
      responseText = enMessages[intent].response;
    } else if (intent === "about_mop") {
      responseText = `${enMessages.about.p1} ${enMessages.about.p2}`;
    } else if (intent === "use_cases") {
      const localResults = searchLocalUseCases(input);
      if (localResults.length > 0) {
        setMessages(prev => [...prev, {
          content: (
            <>
              <div>{enMessages.use_cases.intro}</div>
              <ul>{localResults.map((cs, i) => <li key={i}><strong>{cs.title}</strong>: {cs.description}</li>)}</ul>
            </>
          ),
          sender: "bot",
          text: localResults.map(cs => cs.title).join(", "),
        }]);
        return;
      }

      const apiResults = await fetchUseCasesFromAPI(input);
      if (apiResults.length > 0) {
        setMessages(prev => [...prev, {
          content: (
            <>
              <div>{enMessages.use_cases.intro}</div>
              <ul>{apiResults.slice(0, 5).map((cs: any) => <li key={cs.id}><strong>{cs.name}</strong>: {cs.description}</li>)}</ul>
            </>
          ),
          sender: "bot",
          text: apiResults.map((cs: any) => cs.name).join(", "),
        }]);
        return;
      }

      responseText = enMessages.use_case_prompt.response;
    } else if (intent.startsWith("navigate_")) {
      const routeMap: { [key: string]: string } = {
        navigate_home: "/en/",
        navigate_about: "/en/about",
        navigate_contact: "/en/contact",
        navigate_statistics: "/en/statistics",
        navigate_upload: "/en/upload",
        navigate_sign_up: "/en/signup",
        navigate_log_in: "/en/login"
      };
      router.push(routeMap[intent]);
      responseText = enMessages.navigation[intent.replace("navigate_", "")];
    } else if (intent === "navigate_language") {
      responseText = `${enMessages.language_prompt.response} ${enMessages.language_prompt.languages.join(", ")}`;
    } else if (intent === "faq") {
      window.location.href = enMessages.initial.faq_url;
      return;
    } else {
      const fallback = await fetchUseCasesFromAPI(input);
      responseText = fallback.length > 0
        ? `${enMessages.use_cases.intro} ${fallback.map((cs: any) => `${cs.name}: ${cs.description}`).join(". ")}`
        : enMessages.fallback.response;
    }

    setMessages(prev => [...prev, { content: <>{responseText}</>, sender: "bot", text: responseText }]);
    if (isSpeaking) speakMessage(responseText);
  };

  const handleSend = async () => {
    if (!userInput.trim()) {
      const msg = enMessages.validation.empty_input;
      setMessages(prev => [...prev, { content: <>{msg}</>, sender: "bot", text: msg }]);
      if (isSpeaking) speakMessage(msg);
      return;
    }
    const input = userInput.trim();
    setMessages(prev => [...prev, { content: <>{input}</>, sender: "user", text: input }]);
    setUserInput("");
    await handleCommand(input);
  };

  const handleSendVoiceInput = async (transcript: string) => {
    if (!transcript.trim()) return;
    setMessages(prev => [...prev, { content: <>{transcript}</>, sender: "user", text: transcript }]);
    await handleCommand(transcript);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") handleSend();
  };

  const toggleSpeech = () => {
    if (isSpeaking) stopSpeaking();
    else {
      const lastBot = [...messages].reverse().find(m => m.sender === "bot");
      if (lastBot?.text) speakMessage(lastBot.text);
    }
  };

  return (
    <div className="chatbot fixed bottom-4 right-4 flex flex-col items-end z-[9999]">
      {isOpen && (
        <div className="chat-window p-4 bg-white shadow-lg rounded-lg max-w-xs w-full">
          <div className="flex justify-between items-center mb-2 border-b pb-2">
            <h3 className="text-md font-semibold text-green-600">Melbourne Open Data Assistant</h3>
            <div className="flex space-x-2">
              {speechSupported && (
                <button onClick={toggleSpeech} title="Toggle speech">
                  {isSpeaking ? <BsVolumeUp className="text-green-600" /> : <BsVolumeMute className="text-gray-500" />}
                </button>
              )}
              {recognitionSupported && (
                <button onClick={toggleListening} title="Toggle voice input" className={isListening ? "animate-pulse bg-green-100" : ""}>
                  {isListening ? <BsMicFill className="text-green-600" /> : <BsMicMuteFill className="text-gray-500" />}
                </button>
              )}
            </div>
          </div>
          <div className="messages overflow-auto h-60">
            {messages.map((msg, i) => (
              <div key={i} className={`p-2 my-1 rounded-lg ${msg.sender === "bot" ? "bg-green-50 border-l-4 border-green-500 text-green-800" : "bg-gray-100 text-gray-800 text-right"}`}>
                {msg.content}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
          <div className="input-area flex items-center mt-3 border-t pt-3">
            <input
              type="text"
              className="flex-1 p-2 border rounded-l outline-none focus:ring-2 focus:ring-green-300"
              value={userInput}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder={isListening ? "Listening..." : "Type a message..."}
              disabled={isListening}
            />
            <button onClick={handleSend} className="bg-green-500 text-white p-2 rounded-r hover:bg-green-600">
              <IoSend size={20} />
            </button>
          </div>
          {isListening && <div className="text-xs text-center mt-1 text-green-600">Listening... Speak now!</div>}
        </div>
      )}
      <button onClick={toggleChat} className="text-3xl text-white bg-green-600 rounded-full p-3 hover:bg-green-700 shadow-lg">
        <IoChatbubbleEllipsesSharp />
      </button>
    </div>
  );
};

export default Chatbot;
