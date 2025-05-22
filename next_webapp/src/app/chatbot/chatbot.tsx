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

export interface LiveUseCase {
  id: number;
  name: string;
  description: string;
  filename: string;
  tags: string[];
}

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
  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => setUserInput(event.target.value);
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
    speechSynthRef.current.cancel();
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

  const searchUseCases = async (term: string): Promise<LiveUseCase[]> => {
    const studies = await fetchUseCasesFromAPI(term, "CONTENT");
    return studies.map((uc: any) => ({
      id: uc.id,
      name: uc.name,
      description: uc.description,
      filename: uc.filename,
      tags: uc.tags ?? [],
    }));
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
    const matchedIntent = processInput(input);
    let responseText = "";

    switch (matchedIntent) {
      case "greet":
      case "greet_morning":
      case "greet_afternoon":
      case "greet_evening":
      case "mop_full_form":
      case "project_overview":
      case "contact":
      case "help":
      case "thank_you":
      case "bye":
      case "goodbye":
        responseText = enMessages[matchedIntent].response;
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
          const matches = await searchUseCases(input);
          if (matches.length === 0) {
            responseText = enMessages.use_case_prompt.response;
          } else {
            responseText = enMessages.use_cases.intro;
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
                    <ul className="list-disc pl-4">
                      {matches.slice(0, 5).map((uc) => (
                        <li key={uc.id}>
                          <strong>{uc.name}</strong>: {uc.description}
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
                text: matches.map((m) => m.name).join(", "),
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
      //  Query the live API
        const raw = await searchUseCases(input);          // LiveUseCase[]
        const matches: CaseStudy[] = raw.map((u) => ({
        ...u,
        tags: u.tags ?? []              // supply empty array if the field is missing
    }));

      //  Nothing found → prompt for more detail
    if (matches.length === 0) {
      const prompt = enMessages.use_case_prompt.response;
      setMessages(prev => [...prev, { content: <>{prompt}</>, sender:"bot", text: prompt }]);
      break;
    }

  // Found something → list up to five 
  setMessages(prev => [
    ...prev,
    {
      content: (
        <>
          <div>{enMessages.use_cases.intro}</div>
          <ul className="list-disc pl-4">
            {matches.slice(0,5).map((uc: CaseStudy) => (
              <li key={uc.id}>
                <strong>{uc.name}</strong>: {uc.description}
              </li>
            ))}
          </ul>
        </>
      ),
      sender: "bot",
      text: matches.map((m: CaseStudy) => m.name).join(", ")
    }
  ]);
  break;
}

      case "faq":
        window.location.href = enMessages.initial.faq_url;
        return;

      case "navigate_home":
        router.push("/en/");
        responseText = enMessages.navigation.home;
        break;
      case "navigate_about":
        router.push("/en/about");
        responseText = enMessages.navigation.about;
        break;
      case "navigate_contact":
        router.push("/en/contact");
        responseText = enMessages.navigation.contact;
        break;
      case "navigate_statistics":
        router.push("/en/statistics");
        responseText = enMessages.navigation.statistics;
        break;
      case "navigate_upload":
        router.push("/en/upload");
        responseText = enMessages.navigation.upload;
        break;
      case "navigate_language":
        responseText = `${enMessages.language_prompt.response} ${enMessages.language_prompt.languages.join(", ")}`;
        break;
      case "navigate_sign_up":
        router.push("/en/signup");
        responseText = enMessages.navigation.sign_up;
        break;
      case "navigate_log_in":
        router.push("/en/login");
        responseText = enMessages.navigation.log_in;
        break;

      default:
        const fallbackResults = await fetchUseCasesFromAPI(input);
        if (fallbackResults.length > 0) {
          responseText = `${enMessages.use_cases.intro} ${fallbackResults
            .map((cs: any) => `${cs.name}: ${cs.description}`)
            .join(". ")}`;
        } else {
          responseText = enMessages.fallback.response;
        }
        break;
    }

    if (responseText) {
      setMessages((prev) => [...prev, { content: <>{responseText}</>, sender: "bot", text: responseText }]);
      if (isSpeaking) speakMessage(responseText);
    }
    return responseText;
  };

  const handleSend = async () => {
    if (!userInput.trim()) {
      const emptyInputMessage = enMessages.validation.empty_input;
      setMessages((prev) => [
        ...prev,
        { content: <>{emptyInputMessage}</>, sender: "bot", text: emptyInputMessage },
      ]);
      if (isSpeaking) speakMessage(emptyInputMessage);
      return;
    }
    const trimmedInput = userInput.trim();
    setMessages((prev) => [...prev, { content: <>{userInput}</>, sender: "user", text: userInput }]);
    setUserInput("");
    await handleCommand(trimmedInput);
  };

  const handleSendVoiceInput = async (transcript: string) => {
    if (!transcript.trim()) return;
    setMessages((prev) => [...prev, { content: <>{transcript}</>, sender: "user", text: transcript }]);
    await handleCommand(transcript);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") handleSend();
  };

  const toggleSpeech = () => {
    if (isSpeaking) {
      stopSpeaking();
    } else {
      setIsSpeaking(true);
      const lastBotMessage = [...messages].reverse().find((msg) => msg.sender === "bot");
      if (lastBotMessage?.text) speakMessage(lastBotMessage.text);
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
    <div className="chatbot fixed bottom-4 right-4 flex flex-col items-end z-[9999]">
      {isOpen && (
        <div className="chat-window p-4 bg-white shadow-lg rounded-lg max-w-xs w-full break-words">
          <div className="flex justify-between items-center mb-2 border-b pb-2">
            <h3 className="text-md font-semibold text-green-600">Melbourne Open Data Assistant</h3>
        <div className="chat-window p-4 bg-white shadow-lg rounded-lg max-w-xs w-full">
          <div className="flex justify-between items-center mb-2 border-b pb-2">
            <h3 className="text-md font-semibold text-green-600">
        <div className="chat-window p-4 bg-white shadow-lg rounded-lg max-w-xs w-full text-wrap">
          <div className="flex justify-between items-center mb-2 border-b pb-2 text-wrap">
            <h3 className="text-md font-semibold text-green-600 text-wrap">
              Melbourne Open Data Assistant
            </h3>
            <div className="flex space-x-2">
              {speechSupported && (
                <button onClick={toggleSpeech} className="p-1 rounded" title="Toggle speech">
                  {isSpeaking ? <BsVolumeUp className="text-green-600" /> : <BsVolumeMute className="text-gray-500" />}
                </button>
              )}
              {recognitionSupported && (
                <button onClick={toggleListening} className={`p-1 rounded ${isListening ? "animate-pulse bg-green-100" : ""}`} title="Toggle voice input">
                  {isListening ? <BsMicFill className="text-green-600" /> : <BsMicMuteFill className="text-gray-500" />}
                </button>
              )}
            </div>
          </div>
          <div className="messages overflow-auto h-60">
            {messages.map((msg, i) => (
              <div key={i} className={`message p-2 my-1 rounded-lg ${msg.sender === "bot" ? "bg-green-50 text-green-800 border-l-4 border-green-500" : "bg-gray-100 text-gray-800 text-right"}`}>
                {msg.content}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
          <div className="input-area flex items-center mt-3 border-t pt-3">
            <input type="text" className="flex-1 p-2 border rounded-l outline-none focus:ring-2 focus:ring-green-300" value={userInput} onChange={handleInputChange} onKeyPress={handleKeyPress} placeholder={isListening ? "Listening..." : "Type a message..."} disabled={isListening} />
            <button onClick={handleSend} className="bg-green-500 text-white p-2 rounded-r hover:bg-green-600">
          <div className="input-area flex items-center mt-3 border-t pt-3 text-wrap">
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
          {isListening && <div className="text-xs text-center mt-1 text-green-600">Listening... Speak now!</div>}
        </div>
      )}
      <button onClick={toggleChat} className="toggle-btn text-3xl text-white bg-green-600 rounded-full p-3 hover:bg-green-700 shadow-lg transition-transform hover:scale-105">
        <IoChatbubbleEllipsesSharp />
      </button>
    </div>
  );
};

export default Chatbot;
