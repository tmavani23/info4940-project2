import { useState, useRef, useEffect } from 'react'
import ChatWindow from './components/ChatWindow'
import InputBar from './components/InputBar'
import './App.css'

const DUMMY_RESPONSES = [
  "That's a great question! I'm currently running in demo mode — the Gemini API will be connected soon.",
  "Interesting! Once the Gemini API is integrated, I'll be able to give you a real answer.",
  "I received your message. In the full version, Gemini will analyze and respond to your input.",
  "Got it! This is a placeholder response while the backend is being set up.",
  "Thanks for sharing that. The Gemini model will process this kind of input when connected.",
]

let dummyIndex = 0

function getDummyResponse() {
  const response = DUMMY_RESPONSES[dummyIndex % DUMMY_RESPONSES.length]
  dummyIndex++
  return response
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      id: 0,
      role: 'assistant',
      type: 'text',
      content: 'Hi! I\'m your Gemini-powered assistant. Send me a message, image, or audio clip to get started.',
    },
  ])
  const [isLoading, setIsLoading] = useState(false)
  const idRef = useRef(1)

  function addMessage(msg) {
    const id = idRef.current++
    setMessages((prev) => [...prev, { id, ...msg }])
    return id
  }

  async function handleSend(input) {
    // input: { type: 'text', content: string }
    //      | { type: 'image', content: string (dataURL), file: File }
    //      | { type: 'audio', content: string (dataURL), blob: Blob }
    addMessage({ role: 'user', ...input })
    setIsLoading(true)

    // Simulate network delay
    await new Promise((r) => setTimeout(r, 900 + Math.random() * 600))

    // TODO: replace with Gemini API call
    addMessage({ role: 'assistant', type: 'text', content: getDummyResponse() })
    setIsLoading(false)
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-logo">
          <span className="logo-gem">✦</span>
          <span className="logo-text">P5.js Chat</span>
        </div>
        <span className="demo-badge">Demo Mode</span>
      </header>

      <ChatWindow messages={messages} isLoading={isLoading} />

      <InputBar onSend={handleSend} isLoading={isLoading} />
    </div>
  )
}
