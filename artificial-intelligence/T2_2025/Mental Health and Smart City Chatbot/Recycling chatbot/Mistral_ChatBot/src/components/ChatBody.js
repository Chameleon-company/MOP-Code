import React from "react";
import Message from "./Message";

function ChatBody({ messages, botTyping, panelRef }) {
  return (
    <div className="chat-body" ref={panelRef}>
      {messages.map((msg, idx) => (
        <Message key={idx} text={msg.text} sender={msg.sender} />
      ))}
      {botTyping && <div className="chat-message bot typing">Bot is typing...</div>}
    </div>
  );
}

export default ChatBody;
