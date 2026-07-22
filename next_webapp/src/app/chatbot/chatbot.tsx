"use client";
import React, { useState, useEffect, useRef } from "react";
import { IoChatbubbleEllipsesSharp, IoSend, IoClose } from "react-icons/io5";
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
    if (titleMatch || descMatch || tagMatch) results.push(useCase);
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
          <a href={enMessages.initial.faq_url} className="text-green-300 underline">
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
            <div>
              <p className="mb-1">{enMessages.use_cases.intro}</p>
              <ul className="list-disc list-inside space-y-1">
                {localResults.map((cs, i) => <li key={i}><strong>{cs.title}</strong>: {cs.description}</li>)}
              </ul>
            </div>
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
            <div>
              <p className="mb-1">{enMessages.use_cases.intro}</p>
              <ul className="list-disc list-inside space-y-1">
                {apiResults.slice(0, 5).map((cs: any) => <li key={cs.id}><strong>{cs.name}</strong>: {cs.description}</li>)}
              </ul>
            </div>
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
    <div className="fixed bottom-5 right-5 z-[9999] flex flex-col items-end">
      {isOpen && (
        <div className="chat-window mb-3 flex h-[500px] w-[360px] flex-col overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-2xl dark:border-gray-700 dark:bg-gray-900">
          {/* Header */}
          <div className="flex flex-shrink-0 items-center justify-between bg-green-600 px-4 py-3">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-white/20">
                <IoChatbubbleEllipsesSharp className="text-white" size={17} />
              </div>
              <div>
                <p className="text-sm font-semibold leading-tight text-white">Melbourne Open Data</p>
                <p className="text-xs text-green-100">AI Assistant • Online</p>
              </div>
            </div>
            <div className="flex items-center gap-1">
              {speechSupported && (
                <button
                  onClick={toggleSpeech}
                  title="Toggle speech"
                  className="flex h-8 w-8 items-center justify-center rounded-lg text-white/80 hover:bg-white/15 hover:text-white transition-colors"
                >
                  {isSpeaking ? <BsVolumeUp size={15} /> : <BsVolumeMute size={15} />}
                </button>
              )}
              {recognitionSupported && (
                <button
                  onClick={toggleListening}
                  title="Toggle voice input"
                  className={`flex h-8 w-8 items-center justify-center rounded-lg text-white/80 hover:bg-white/15 hover:text-white transition-colors ${isListening ? "bg-white/20 animate-pulse" : ""}`}
                >
                  {isListening ? <BsMicFill size={15} /> : <BsMicMuteFill size={15} />}
                </button>
              )}
              <button
                onClick={toggleChat}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-white/80 hover:bg-white/15 hover:text-white transition-colors"
                aria-label="Close chat"
              >
                <IoClose size={18} />
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
            {messages.map((msg, i) => (
              <div key={i} className={`flex items-end gap-2 ${msg.sender === "user" ? "justify-end" : "justify-start"}`}>
                {msg.sender === "bot" && (
                  <div className="mb-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-green-100 text-green-600 dark:bg-green-900/40 dark:text-green-400">
                    <IoChatbubbleEllipsesSharp size={13} />
                  </div>
                )}
                <div
                  className={`message max-w-[78%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${
                    msg.sender === "user"
                      ? "rounded-br-sm bg-green-600 text-white"
                      : "rounded-bl-sm bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-100"
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="flex-shrink-0 border-t border-gray-100 px-3 py-3 dark:border-gray-700">
            {isListening && (
              <p className="mb-2 text-center text-xs font-medium text-green-600 dark:text-green-400 animate-pulse">
                Listening... Speak now!
              </p>
            )}
            <div className="flex items-center gap-2 rounded-xl border border-gray-200 bg-gray-50 px-3 py-2 focus-within:border-green-400 focus-within:ring-1 focus-within:ring-green-300 transition-all dark:border-gray-700 dark:bg-gray-800">
              {recognitionSupported && (
                <button
                  onClick={toggleListening}
                  title="Voice input"
                  className={`flex-shrink-0 text-gray-400 hover:text-green-600 dark:hover:text-green-400 transition-colors ${isListening ? "text-green-600 dark:text-green-400 animate-pulse" : ""}`}
                >
                  {isListening ? <BsMicFill size={16} /> : <BsMicMuteFill size={16} />}
                </button>
              )}
              <input
                type="text"
                className="flex-1 bg-transparent text-sm text-gray-800 outline-none placeholder:text-gray-400 dark:text-gray-100"
                value={userInput}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder={isListening ? "Listening..." : "Type a message..."}
                disabled={isListening}
              />
              <button
                onClick={handleSend}
                className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-green-600 text-white hover:bg-green-700 transition-colors"
                aria-label="Send message"
              >
                <IoSend size={14} />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Toggle Button */}
      <button
        onClick={toggleChat}
        className="flex h-14 w-14 items-center justify-center rounded-full bg-green-600 text-white shadow-xl hover:bg-green-700 hover:scale-105 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
        aria-label={isOpen ? "Close chat" : "Open chat"}
      >
        {isOpen ? <IoClose size={22} /> : <IoChatbubbleEllipsesSharp size={22} />}
      </button>
    </div>
  );
};

export default Chatbot;