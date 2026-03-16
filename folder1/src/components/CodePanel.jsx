import { useState } from 'react'
import './CodePanel.css'

export default function CodePanel({ code, onOpenPreview }) {
  const [copied, setCopied] = useState(false)

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // fallback for browsers without clipboard API
    }
  }

  return (
    <aside className="code-panel">
      <div className="code-panel-header">
        <div className="code-panel-title">
          <span className="code-panel-icon">{ }</span>
          <span>p5.js Code</span>
        </div>
        <div className="code-panel-actions">
          <button className="cp-btn" onClick={handleCopy}>
            {copied ? '✓ Copied' : 'Copy'}
          </button>
          <button className="cp-btn cp-btn-primary" onClick={onOpenPreview}>
            ▶ Preview
          </button>
        </div>
      </div>

      <div className="code-panel-body">
        <pre className="code-block"><code>{code}</code></pre>
      </div>
    </aside>
  )
}
