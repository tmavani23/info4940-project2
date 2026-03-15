import { parseMessageContent, simpleMarkdown } from '../utils'
import './ChatWindow.css'

export default function MessageBubble({ message }) {
  const { role, type, content, file } = message

  return (
    <div className={`message ${role}`}>
      <div className={`avatar ${role === 'assistant' ? 'assistant-avatar' : 'user-avatar'}`}>
        {role === 'assistant' ? '✦' : '👤'}
      </div>

      <div className="bubble">
        {type === 'text' && <TextContent content={content} />}

        {type === 'image' && (
          <>
            <img className="chat-image" src={content} alt="uploaded" />
            {file?.name && <p className="image-caption">{file.name}</p>}
          </>
        )}

        {type === 'audio' && <audio controls src={content} />}
      </div>
    </div>
  )
}

function TextContent({ content }) {
  const parts = parseMessageContent(content)

  return (
    <>
      {parts.map((part, i) =>
        part.type === 'code' ? (
          <pre key={i} className="inline-code-block">
            <code>{part.content}</code>
          </pre>
        ) : (
          <span
            key={i}
            dangerouslySetInnerHTML={{ __html: simpleMarkdown(part.content) }}
          />
        )
      )}
    </>
  )
}
