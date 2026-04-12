import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './App.css'

type Message = { from: 'user' | 'bot'; text: string }

function App() {
  const [history, setHistory] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  // auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history])

  const sendMessage = async () => {
    if (!input.trim()) return

    // add user message
    setHistory(h => [...h, { from: 'user', text: input }])
    const payload = { text: input }
    setInput('')

    try {
      const res = await axios.post('/api/chat', payload)
      const botText = res.data.response as string

      // add bot response
      setHistory(h => [...h, { from: 'bot', text: botText }])
    } catch (err) {
      console.error(err)
      setHistory(h => [
        ...h,
        { from: 'bot', text: '❗️Error talking to server' },
      ])
    }
  }

  return (
    <div className="chat-container">
      <div className="chat-history">
        {history.map((m, i) => (
          <div key={i} className={`message ${m.from}`}>
            {m.text}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="input-bar">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && sendMessage()}
          placeholder="Type a message…"
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  )
}

export default App
