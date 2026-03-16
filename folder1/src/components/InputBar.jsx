import { useState, useRef } from 'react'
import './InputBar.css'

export default function InputBar({ onSend, isLoading }) {
  const [text, setText] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [recordingSeconds, setRecordingSeconds] = useState(0)
  const [preview, setPreview] = useState(null) // { type, content, file?, blob? }

  const fileInputRef = useRef(null)
  const textareaRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const timerRef = useRef(null)

  // --- image upload ---
  function handleImageChange(e) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      setPreview({ type: 'image', content: ev.target.result, file })
    }
    reader.readAsDataURL(file)
    e.target.value = ''
  }

  function clearPreview() {
    setPreview(null)
  }

  // --- audio recording ---
  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      audioChunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data)
      }

      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        const url = URL.createObjectURL(blob)
        setPreview({ type: 'audio', content: url, blob })
        clearInterval(timerRef.current)
        setRecordingSeconds(0)
      }

      recorder.start()
      mediaRecorderRef.current = recorder
      setIsRecording(true)
      setRecordingSeconds(0)
      timerRef.current = setInterval(() => setRecordingSeconds((s) => s + 1), 1000)
    } catch {
      alert('Microphone access is required to record audio.')
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop()
    setIsRecording(false)
  }

  function toggleRecording() {
    if (isRecording) stopRecording()
    else startRecording()
  }

  // --- send ---
  function handleSend() {
    if (isLoading) return

    if (preview) {
      onSend(preview)
      setPreview(null)
      setText('')
      return
    }

    const trimmed = text.trim()
    if (!trimmed) return
    onSend({ type: 'text', content: trimmed })
    setText('')
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const canSend = !isLoading && !isRecording && (text.trim().length > 0 || preview !== null)

  const fmt = (s) => `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`

  return (
    <div className="input-area">
      {/* attachment preview strip */}
      {preview && (
        <div className="preview-strip">
          {preview.type === 'image' && (
            <div className="preview-item">
              <img src={preview.content} alt="preview" className="preview-image" />
              <span className="preview-label">{preview.file?.name ?? 'Image'}</span>
            </div>
          )}
          {preview.type === 'audio' && (
            <div className="preview-item">
              <audio controls src={preview.content} className="preview-audio" />
            </div>
          )}
          <button className="remove-preview" onClick={clearPreview} title="Remove">✕</button>
        </div>
      )}

      <div className="input-bar">
        {/* image upload */}
        <button
          className="icon-btn"
          title="Attach image"
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading || isRecording}
        >
          <ImageIcon />
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={handleImageChange}
        />

        {/* audio record */}
        <button
          className={`icon-btn ${isRecording ? 'recording' : ''}`}
          title={isRecording ? 'Stop recording' : 'Record audio'}
          onClick={toggleRecording}
          disabled={isLoading || preview?.type === 'image'}
        >
          {isRecording ? <StopIcon /> : <MicIcon />}
        </button>

        {isRecording && (
          <span className="recording-timer">{fmt(recordingSeconds)}</span>
        )}

        {/* text input */}
        <textarea
          ref={textareaRef}
          className="text-input"
          placeholder="Message Gemini…"
          value={text}
          onChange={(e) => {
            setText(e.target.value)
            const ta = textareaRef.current
            if (ta) {
              ta.style.height = 'auto'
              ta.style.height = Math.min(ta.scrollHeight, 140) + 'px'
            }
          }}
          onKeyDown={handleKeyDown}
          disabled={isLoading || isRecording || preview !== null}
          rows={1}
        />

        {/* send */}
        <button
          className={`send-btn ${canSend ? 'active' : ''}`}
          onClick={handleSend}
          disabled={!canSend}
          title="Send"
        >
          <SendIcon />
        </button>
      </div>

      <p className="input-hint">Enter to send · Shift+Enter for newline</p>
    </div>
  )
}

// --- inline SVG icons ---
function MicIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a3 3 0 0 1 3 3v7a3 3 0 0 1-6 0V5a3 3 0 0 1 3-3z"/>
      <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
      <line x1="12" y1="19" x2="12" y2="23"/>
      <line x1="8" y1="23" x2="16" y2="23"/>
    </svg>
  )
}

function StopIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <rect x="4" y="4" width="16" height="16" rx="2"/>
    </svg>
  )
}

function ImageIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2"/>
      <circle cx="8.5" cy="8.5" r="1.5"/>
      <polyline points="21 15 16 10 5 21"/>
    </svg>
  )
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13"/>
      <polygon points="22 2 15 22 11 13 2 9 22 2"/>
    </svg>
  )
}
