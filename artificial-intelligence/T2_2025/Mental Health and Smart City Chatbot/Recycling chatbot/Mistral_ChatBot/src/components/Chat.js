import React, { useState, useRef, useEffect } from "react";
import ChatHeader from "./ChatHeader";
import ChatBody from "./ChatBody";
import ChatFooter from "./ChatFooter";

function Chat() {
  const [messages, setMessages] = useState([]); 
  const [botTyping, setBotTyping] = useState(false);
  const panelRef = useRef(null);

  const appendMessage = (text, sender) => {
    setMessages((prev) => [...prev, { text, sender }]);
  };

  const sendMessage = async (input) => {
    const trimmed = input.trim();
    if (!trimmed) return;

    appendMessage(trimmed, "user");
    setBotTyping(true);

    try {
      const res = await fetch("http://127.0.0.1:5000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed })
      });
      const data = await res.json();

      setTimeout(() => {
        appendMessage(data.answer, "bot");
        setBotTyping(false);
      }, 500); 
    } catch (err) {
      console.error(err);
      appendMessage("Sorry, something went wrong.", "bot");
      setBotTyping(false);
    }
  };

  useEffect(() => {
    if (panelRef.current) {
      panelRef.current.scrollTop = panelRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="chat-container">
      <div className="chat-panel">
        <ChatHeader />
        <ChatBody messages={messages} botTyping={botTyping} panelRef={panelRef} />
        <ChatFooter onSend={sendMessage} />
      </div>
    </div>
  );
}

export default Chat;
