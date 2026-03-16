import { useState, useRef } from 'react'
import ChatWindow from './components/ChatWindow'
import InputBar from './components/InputBar'
import CodePanel from './components/CodePanel'
import PreviewPage from './components/PreviewPage'
import { extractP5Code } from './utils'
import './App.css'

// ---------------------------------------------------------------------------
// Dummy responses — some include p5.js sketches to model the full interaction
// ---------------------------------------------------------------------------
const DUMMY_RESPONSES = [
  `Here's a p5.js bouncing ball animation to get us started:

\`\`\`javascript
let x, y, vx, vy, r;

function setup() {
  createCanvas(windowWidth, windowHeight);
  x = width / 2;
  y = height / 2;
  vx = 3;
  vy = 2.5;
  r = 24;
}

function draw() {
  background(15, 15, 25);
  x += vx;
  y += vy;
  if (x < r || x > width - r) vx *= -1;
  if (y < r || y > height - r) vy *= -1;
  noStroke();
  fill(100, 180, 255);
  ellipse(x, y, r * 2);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}
\`\`\`

Switch to the **Preview** tab to see it in action!`,

  `Got it! This is a placeholder response while the Gemini API is being connected.`,

  `Here's a colorful spiral animation using HSB color mode:

\`\`\`javascript
function setup() {
  createCanvas(windowWidth, windowHeight);
  colorMode(HSB, 360, 100, 100, 100);
}

function draw() {
  background(240, 30, 10, 15);
  translate(width / 2, height / 2);
  let t = frameCount * 0.015;
  for (let i = 0; i < 8; i++) {
    let angle = t + (TWO_PI / 8) * i;
    let r = 120 * sin(t * 0.7 + i);
    let x = r * cos(angle);
    let y = r * sin(angle);
    fill((frameCount * 2 + i * 45) % 360, 80, 100, 80);
    noStroke();
    circle(x, y, 14);
  }
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}
\`\`\`

Open the **Preview** tab to watch it animate!`,

  `I received your message. In the full version, Gemini will analyze and respond to your input.`,

  `Here's a Perlin noise wave visualization:

\`\`\`javascript
let t = 0;

function setup() {
  createCanvas(windowWidth, windowHeight);
  colorMode(HSB, 360, 100, 100, 100);
}

function draw() {
  background(220, 30, 12);
  noFill();
  strokeWeight(2);
  for (let w = 0; w < 6; w++) {
    stroke((200 + w * 22) % 360, 70, 100, 80 - w * 10);
    beginShape();
    for (let x = 0; x <= width; x += 6) {
      let n = noise(x * 0.003, w * 0.6 + t);
      let y = map(n, 0, 1, height * 0.2, height * 0.8);
      vertex(x, y);
    }
    endShape();
  }
  t += 0.008;
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}
\`\`\`

Check out the **Preview** tab!`,

  `Thanks for sharing that. The Gemini model will process this kind of input when connected.`,
]

let dummyIndex = 0

export default function App() {
  const [messages, setMessages] = useState([
    {
      id: 0,
      role: 'assistant',
      type: 'text',
      content: "Hi! I'm your Gemini-powered assistant. Send me a message, image, or audio clip to get started.",
    },
  ])
  const [isLoading, setIsLoading] = useState(false)
  const [activePage, setActivePage] = useState('chat') // 'chat' | 'preview'
  const [p5Code, setP5Code] = useState(null)
  const idRef = useRef(1)

  function addMessage(msg) {
    const id = idRef.current++
    setMessages((prev) => [...prev, { id, ...msg }])
  }

  async function handleSend(input) {
    addMessage({ role: 'user', ...input })
    setIsLoading(true)

    await new Promise((r) => setTimeout(r, 900 + Math.random() * 600))

    const responseText = DUMMY_RESPONSES[dummyIndex % DUMMY_RESPONSES.length]
    dummyIndex++

    addMessage({ role: 'assistant', type: 'text', content: responseText })

    // Detect p5.js code and surface it in the code panel
    const code = extractP5Code(responseText)
    if (code) setP5Code(code)

    setIsLoading(false)
  }

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-logo">
          <span className="logo-gem">✦</span>
          <span className="logo-text">P5.js Chat</span>
        </div>

        <nav className="tab-nav">
          <button
            className={`tab-btn ${activePage === 'chat' ? 'active' : ''}`}
            onClick={() => setActivePage('chat')}
          >
            Chat
          </button>
          <button
            className={`tab-btn ${activePage === 'preview' ? 'active' : ''}`}
            onClick={() => setActivePage('preview')}
          >
            Preview
            {p5Code && activePage !== 'preview' && <span className="tab-dot" />}
          </button>
        </nav>

        <span className="demo-badge">Demo Mode</span>
      </header>

      {/* ── Chat page ── */}
      {activePage === 'chat' && (
        <div className="chat-layout">
          <div className="chat-pane">
            <ChatWindow messages={messages} isLoading={isLoading} />
            <InputBar onSend={handleSend} isLoading={isLoading} />
          </div>

          {p5Code && (
            <CodePanel
              code={p5Code}
              onOpenPreview={() => setActivePage('preview')}
            />
          )}
        </div>
      )}

      {/* ── Preview page ── */}
      {activePage === 'preview' && (
        <PreviewPage code={p5Code} />
      )}
    </div>
  )
}
