import './ChatWindow.css'

export default function MessageBubble({ message }) {
  const { role, type, content, file } = message

  return (
    <div className={`message ${role}`}>
      <div className={`avatar ${role === 'assistant' ? 'assistant-avatar' : 'user-avatar'}`}>
        {role === 'assistant' ? '✦' : '👤'}
      </div>

      <div className="bubble">
        {type === 'text' && <span>{content}</span>}

        {type === 'image' && (
          <>
            <img className="chat-image" src={content} alt="uploaded" />
            {file?.name && (
              <p className="image-caption">{file.name}</p>
            )}
          </>
        )}

        {type === 'audio' && (
          <audio controls src={content} />
        )}
      </div>
    </div>
  )
}
