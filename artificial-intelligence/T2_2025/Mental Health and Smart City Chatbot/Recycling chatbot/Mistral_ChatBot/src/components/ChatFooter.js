import React, { useState } from "react";

function ChatFooter({ onSend }) {
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (input.trim()) {
      onSend(input);
      setInput("");
    }
  };

  return (
    <div className="chat-footer">
      <input
        type="text"
        placeholder="Enter your message..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyPress={(e) => { if (e.key === "Enter") handleSend(); }}
        autoFocus
      />
      <button onClick={handleSend}>Send</button>
    </div>
  );
}

export default ChatFooter;
