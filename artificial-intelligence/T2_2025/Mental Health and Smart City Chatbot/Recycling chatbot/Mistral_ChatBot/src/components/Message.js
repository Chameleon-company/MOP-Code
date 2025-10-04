import React from "react";

function Message({ text, sender }) {
  const linkify = (text) => {
    const urlRegex = /(https?:\/\/[^\s]+)/;
    return text.split(urlRegex).map((part, i) =>
      urlRegex.test(part) ? (
        <a key={i} href={part} target="_blank" rel="noopener noreferrer">
          {part}
        </a>
      ) : (
        part
      )
    );
  };

  return (
    <div className={`chat-message ${sender}`}>
      {linkify(text)}
    </div>
  );
}

export default Message;
