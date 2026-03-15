import './PreviewPage.css'

function buildSrcdoc(code) {
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { overflow: hidden; background: #0f0f10; }
    canvas { display: block; }
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"></script>
</head>
<body>
  <script>
${code}
  </script>
</body>
</html>`
}

export default function PreviewPage({ code }) {
  if (!code) {
    return (
      <div className="preview-empty">
        <div className="preview-empty-icon">✦</div>
        <p className="preview-empty-title">No sketch yet</p>
        <p className="preview-empty-hint">
          Chat with the assistant to generate p5.js code, then come back here to see it run.
        </p>
      </div>
    )
  }

  return (
    <div className="preview-page">
      <iframe
        key={code}
        title="p5.js Preview"
        srcDoc={buildSrcdoc(code)}
        className="preview-iframe"
        sandbox="allow-scripts"
      />
    </div>
  )
}
