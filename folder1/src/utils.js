/**
 * Extract the first p5.js code block from a text string.
 * Returns the raw code string, or null if none found.
 */
export function extractP5Code(text) {
  if (typeof text !== 'string') return null
  const regex = /```(?:javascript|js|p5)?\n([\s\S]*?)```/g
  let match
  while ((match = regex.exec(text)) !== null) {
    const code = match[1].trim()
    if (
      code.includes('function setup') ||
      code.includes('function draw') ||
      code.includes('createCanvas')
    ) {
      return code
    }
  }
  return null
}

/**
 * Split a message string into alternating text / code-block segments.
 * Returns an array of { type: 'text' | 'code', content: string }.
 */
export function parseMessageContent(text) {
  if (typeof text !== 'string') return [{ type: 'text', content: '' }]
  const parts = []
  const regex = /```(?:[a-z]*)?\n([\s\S]*?)```/g
  let lastIndex = 0
  let match

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: 'text', content: text.slice(lastIndex, match.index) })
    }
    parts.push({ type: 'code', content: match[1].trim() })
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < text.length) {
    parts.push({ type: 'text', content: text.slice(lastIndex) })
  }

  return parts.length > 0 ? parts : [{ type: 'text', content: text }]
}

/**
 * Minimal markdown: **bold** and newlines only.
 * Returns an HTML string safe to use with dangerouslySetInnerHTML.
 */
export function simpleMarkdown(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>')
}
