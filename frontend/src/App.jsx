import { useState, useRef, useCallback, useEffect } from "react";
import "./App.css";

const API = "http://localhost:8000";
const FILE_SIZE_WARNING_MB = 10;

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// ── Components ─────────────────────────────────────────────────────────────

function Splash() {
  return (
    <div className="splash-screen">
      <div className="splash-content">
        <div className="splash-logo">🌿</div>
        <h1 className="splash-title">AI Vaidya</h1>
        <p className="splash-subtitle">Awakening Knowledge</p>
      </div>
    </div>
  );
}

function FileItem({ file, onRemove, isRemoving }) {
  return (
    <div className={`file-item ${isRemoving ? "removing" : ""}`}>
      <div className="file-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="16" y1="13" x2="8" y2="13" />
          <line x1="16" y1="17" x2="8" y2="17" />
          <polyline points="10 9 9 9 8 9" />
        </svg>
      </div>
      <div className="file-info">
        <span className="file-name" title={file.name}>{file.name}</span>
        <span className="file-meta">
          {formatBytes(file.size)}&nbsp;·&nbsp;
          <span className="file-badge">
            <span className="badge-dot" /> Indexed
          </span>
        </span>
      </div>
      <button
        className="btn-remove"
        onClick={() => onRemove(file.name)}
        disabled={isRemoving}
        title="Remove file"
        aria-label={`Remove ${file.name}`}
      >
        {isRemoving ? (
          <span className="spinner-xs" />
        ) : (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="3 6 5 6 21 6" />
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
            <path d="M10 11v6M14 11v6" />
            <path d="M9 6V4h6v2" />
          </svg>
        )}
      </button>
    </div>
  );
}

function AskStatus({ status }) {
  const steps = [
    { key: "thinking",   label: "Understanding your question…",   icon: "🔍" },
    { key: "retrieving", label: "Searching Ayurvedic knowledge…", icon: "📚" },
    { key: "generating", label: "Composing answer with sources…", icon: "✍️"  },
  ];
  const activeIndex = steps.findIndex(s => s.key === status);
  if (!status || status === "done" || status === "error") return null;
  return (
    <div className="ask-status-bar" aria-live="polite">
      <div className="ask-status-steps">
        {steps.map((step, i) => {
          const state = i < activeIndex ? "done" : i === activeIndex ? "active" : "pending";
          return (
            <div key={step.key} className={`ask-step ask-step--${state}`}>
              <div className="ask-step-icon">
                {state === "done" ? (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                ) : state === "active" ? (
                  <span className="spinner-step" />
                ) : (
                  <span>{step.icon}</span>
                )}
              </div>
              <span className="ask-step-label">{step.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Toast({ message, type, onClose }) {
  useEffect(() => {
    const timer = setTimeout(onClose, 4000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className={`toast toast--${type}`} role="alert">
      <span className="toast-icon">
        {type === "success" ? "✨" : type === "error" ? "❌" : "ℹ️"}
      </span>
      <span className="toast-msg">{message}</span>
      <button className="toast-close" onClick={onClose} aria-label="Close">×</button>
    </div>
  );
}

function UploadProgressBar({ progress, status }) {
  if (status !== "uploading" && status !== "processing") return null;

  const isProcessing = status === "processing";
  const displayPercent = isProcessing ? 100 : Math.round(progress);

  return (
    <div className="upload-progress-section">
      <div className="upload-progress-bar">
        <div
          className={`upload-progress-fill ${isProcessing ? "processing-pulse" : ""}`}
          style={{ width: `${displayPercent}%` }}
        />
      </div>
      <div className="upload-progress-info">
        {isProcessing ? (
          <>
            <span className="spinner-xs" />
            <span>Processing & indexing PDF…</span>
          </>
        ) : (
          <>
            <span className="progress-percent">{displayPercent}%</span>
            <span>Uploading…</span>
          </>
        )}
      </div>
    </div>
  );
}

function SampleQuestions({ questions, onSelect, disabled }) {
  if (!questions || questions.length === 0) return null;

  return (
    <div className="sample-questions stagger-3">
      <div className="sample-questions-label">
        <span>💡</span> Suggested Insights:
      </div>
      <div className="sample-questions-list">
        {questions.map((q, i) => (
          <button
            key={i}
            className="sample-question-btn"
            onClick={() => onSelect(q)}
            disabled={disabled}
            title={q}
          >
            <span className="sq-icon">→</span>
            <span className="sq-text">{q}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────

export default function App() {
  const [showSplash, setShowSplash]       = useState(true);
  
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [uploadState, setUploadState]     = useState("idle");  // idle | uploading | processing | error
  const [uploadMsg, setUploadMsg]         = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [removingFile, setRemovingFile]   = useState(null);
  const [dragOver, setDragOver]           = useState(false);
  const [fileSizeWarning, setFileSizeWarning] = useState(null);

  const [question, setQuestion] = useState("");
  const [askStatus, setAskStatus] = useState(null);
  const [answer, setAnswer]     = useState(null);
  const [sources, setSources]   = useState([]);
  const [askError, setAskError] = useState("");

  const [sampleQuestions, setSampleQuestions] = useState([]);
  const [toast, setToast] = useState(null);

  const fileInputRef = useRef(null);
  const answerRef    = useRef(null);
  const xhrRef       = useRef(null);

  // Splash Screen Logic
  useEffect(() => {
    const timer = setTimeout(() => {
      setShowSplash(false);
    }, 2800); // Wait for splash animation to finish
    return () => clearTimeout(timer);
  }, []);

  // Fetch sample questions when files change
  useEffect(() => {
    if (uploadedFiles.length > 0) {
      fetchSampleQuestions();
    } else {
      setSampleQuestions([]);
    }
  }, [uploadedFiles]);

  async function fetchSampleQuestions() {
    try {
      const res = await fetch(`${API}/sample-questions`);
      const data = await res.json();
      setSampleQuestions(data.questions || []);
    } catch (_) {
      setSampleQuestions([]);
    }
  }

  function showToast(message, type = "success") {
    setToast({ message, type, id: Date.now() });
  }

  const handleFiles = useCallback(async (fileList) => {
    const file = fileList[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadMsg("Only PDF files are accepted.");
      setUploadState("error");
      return;
    }

    const sizeMB = file.size / (1024 * 1024);
    if (sizeMB > FILE_SIZE_WARNING_MB) {
      setFileSizeWarning(
        `This file is ${sizeMB.toFixed(1)} MB. Large files may take longer to process.`
      );
    } else {
      setFileSizeWarning(null);
    }

    setUploadState("uploading");
    setUploadProgress(0);
    setUploadMsg(`Uploading ${file.name}…`);

    const form = new FormData();
    form.append("file", file);

    try {
      const uploadResult = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhrRef.current = xhr;

        xhr.upload.addEventListener("progress", (e) => {
          if (e.lengthComputable) {
            const pct = (e.loaded / e.total) * 100;
            setUploadProgress(pct);
          }
        });

        xhr.addEventListener("load", () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              resolve(JSON.parse(xhr.responseText));
            } catch {
              resolve({});
            }
          } else {
            try {
              const err = JSON.parse(xhr.responseText);
              reject(new Error(err.detail || `Upload failed (${xhr.status})`));
            } catch {
              reject(new Error(`Upload failed (${xhr.status})`));
            }
          }
        });

        xhr.addEventListener("error", () => reject(new Error("Network error during upload")));
        xhr.addEventListener("abort", () => reject(new Error("Upload cancelled")));

        xhr.open("POST", `${API}/upload`);
        xhr.send(form);
      });

      xhrRef.current = null;

      setUploadState("processing");
      setUploadMsg("Processing & building knowledge index…");
      setUploadProgress(100);

      await pollFileStatus(file.name);

      setUploadedFiles(prev => {
        const exists = prev.find(f => f.name === file.name);
        if (exists) return prev.map(f => f.name === file.name ? { name: file.name, size: file.size } : f);
        return [...prev, { name: file.name, size: file.size }];
      });
      setUploadState("idle");
      setUploadMsg("");
      setUploadProgress(0);
      setFileSizeWarning(null);
      showToast(`${file.name} uploaded and indexed successfully!`, "success");

    } catch (err) {
      xhrRef.current = null;
      setUploadState("error");
      setUploadMsg(err.message || "Something went wrong.");
      setUploadProgress(0);
    }
  }, []);

  async function pollFileStatus(filename, maxMs = 120000) {
    const start = Date.now();
    while (Date.now() - start < maxMs) {
      await new Promise(r => setTimeout(r, 1500));
      try {
        const res  = await fetch(`${API}/files/${encodeURIComponent(filename)}/status`);
        const data = await res.json();
        if (data.status === "done" || data.status === "ready") return;
        if (data.status === "error") {
          throw new Error(data.error || "Processing failed on server.");
        }
      } catch (e) {
        if (e.message && !e.message.includes("fetch")) throw e;
      }
    }
    throw new Error("Processing timed out. Please try again.");
  }

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  async function handleRemove(filename) {
    setRemovingFile(filename);
    try {
      const res = await fetch(`${API}/pdfs/${encodeURIComponent(filename)}`, { method: "DELETE" });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail || "Delete failed");
      }
      setUploadedFiles(prev => prev.filter(f => f.name !== filename));
      setAnswer(null);
      setSources([]);
      setAskError("");
      showToast(`${filename} removed from knowledge base.`, "success");
    } catch (err) {
      showToast(`Could not remove file: ${err.message}`, "error");
    } finally {
      setRemovingFile(null);
    }
  }

  async function handleAsk() {
    if (!question.trim()) return;
    if (uploadedFiles.length === 0) { setAskError("Please upload a PDF first."); return; }
    setAnswer(null);
    setSources([]);
    setAskError("");
    setAskStatus("thinking");
    await delay(700);
    setAskStatus("retrieving");
    let data;
    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim() }),
      });
      data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Request failed");
    } catch (err) {
      setAskStatus("error");
      setAskError(err.message || "Failed to get an answer.");
      return;
    }
    setAskStatus("generating");
    await delay(500);
    setAnswer(data.answer);
    setSources(data.sources || []);
    setAskStatus("done");
    setTimeout(() => answerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
  }

  function handleSampleQuestionSelect(q) {
    setQuestion(q);
    setTimeout(() => {
      if (uploadedFiles.length > 0) {
        setAnswer(null);
        setSources([]);
        setAskError("");
        performAsk(q);
      }
    }, 300);
  }

  async function performAsk(q) {
    setAskStatus("thinking");
    await delay(700);
    setAskStatus("retrieving");
    let data;
    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q.trim() }),
      });
      data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Request failed");
    } catch (err) {
      setAskStatus("error");
      setAskError(err.message || "Failed to get an answer.");
      return;
    }
    setAskStatus("generating");
    await delay(500);
    setAnswer(data.answer);
    setSources(data.sources || []);
    setAskStatus("done");
    setTimeout(() => answerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
  }

  function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

  const kbReady   = uploadedFiles.length > 0;
  const isAsking  = askStatus && askStatus !== "done" && askStatus !== "error";

  if (showSplash) {
    return <Splash />;
  }

  return (
    <>
      <div className="dynamic-bg" />
      <div className="page content-reveal">
        {toast && (
          <Toast
            key={toast.id}
            message={toast.message}
            type={toast.type}
            onClose={() => setToast(null)}
          />
        )}

        {/* Header */}
        <header className="header stagger-1">
          <div className="logo-container">
            <span className="logo-leaf glow">🌿</span>
          </div>
          <h1>AI Vaidya</h1>
          <p>Ayurvedic Knowledge Assistant powered by Semantic AI</p>
        </header>

        {/* Feature Pills */}
        <div className="features stagger-2">
          {[
            { icon: "📚", title: "Ayurveda PDFs",  sub: "Uploaded knowledge base" },
            { icon: "🧠", title: "Semantic Search", sub: "FAISS + BM25 hybrid"     },
            { icon: "🌱", title: "Grounded AI",     sub: "Answers from your texts" },
          ].map((f, i) => (
            <div className={`feature-pill pill-${i+1}`} key={f.title}>
              <span className="pill-icon">{f.icon}</span>
              <strong>{f.title}</strong>
              <span>{f.sub}</span>
            </div>
          ))}
        </div>

        {/* Upload Card */}
        <div className="card glass-card stagger-3">
          <h2 className="section-title">
            <span className="icon-glow">📄</span> Upload Ayurveda PDF
          </h2>
          <div
            className={`upload-zone ${dragOver ? "drag-over" : ""} ${uploadState === "error" ? "zone-error" : ""}`}
            onDragOver={e => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              style={{ display: "none" }}
              onChange={e => { handleFiles(e.target.files); e.target.value = ""; }}
            />
            <span className={`upload-big-icon ${uploadState === "uploading" || uploadState === "processing" ? "pulse-anim" : ""}`}>
              {uploadState === "uploading" || uploadState === "processing" ? "⏳" : "☁️"}
            </span>
            <p className="upload-text">{dragOver ? "Drop your PDF here" : "Drag & drop a PDF or click to browse"}</p>
            <small className="upload-subtext">Ayurvedic texts, Charaka Samhita, research papers…</small>
            <button
              className="btn-primary btn-glow"
              onClick={e => { e.stopPropagation(); fileInputRef.current?.click(); }}
              disabled={uploadState === "uploading" || uploadState === "processing"}
            >
              {uploadState === "uploading" ? <><span className="spinner-xs spinner-gold" /> Uploading…</>
              : uploadState === "processing" ? <><span className="spinner-xs spinner-gold" /> Processing…</>
              : <><span>🌿</span> Choose PDF</>}
            </button>
          </div>

          {fileSizeWarning && (
            <div className="file-size-warning">
              <span>⚠️</span>
              <span>{fileSizeWarning}</span>
            </div>
          )}

          <UploadProgressBar progress={uploadProgress} status={uploadState} />

          {uploadMsg && uploadState === "error" && (
            <div className="upload-feedback feedback-error">
              <span>⚠️</span>
              <span>{uploadMsg}</span>
            </div>
          )}
        </div>

        {/* File Manager */}
        {uploadedFiles.length > 0 && (
          <div className="card glass-card file-manager-card stagger-4">
            <div className="file-manager-header">
              <h2 className="section-title" style={{ margin: 0 }}>
                <span className="icon-glow">🗂️</span> Knowledge Base
              </h2>
              <div className="kb-badge">
                <span className="badge-dot pulse" />
                {uploadedFiles.length} file{uploadedFiles.length !== 1 ? "s" : ""} indexed
              </div>
            </div>

            <div className="kb-status-strip">
              <svg className="status-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                <polyline points="22 4 12 14.01 9 11.01" />
              </svg>
              Vectorstore ready · Hybrid retrieval (FAISS + BM25) active
            </div>

            <div className="file-list">
              {uploadedFiles.map((file, i) => (
                <div key={file.name} style={{animationDelay: `${i * 0.1}s`}} className="file-item-wrapper">
                  <FileItem
                    file={file}
                    onRemove={handleRemove}
                    isRemoving={removingFile === file.name}
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Q&A Section */}
        <div className="card glass-card stagger-5">
          <h2 className="section-title"><span className="icon-glow">🧠</span> Ask a Question</h2>
          <div className={`qa-input-wrap ${!kbReady ? "qa-disabled" : ""}`}>
            <textarea
              className="qa-textarea"
              placeholder={kbReady ? "e.g. What are the three doshas in Ayurveda?" : "Upload a PDF above to unlock Q&A…"}
              value={question}
              onChange={e => setQuestion(e.target.value)}
              disabled={!kbReady}
              rows={3}
              onKeyDown={e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleAsk(); }}
            />
            <div className="qa-hint">Press Ctrl + Enter to ask</div>
          </div>

          <button
            className="btn-ask btn-glow"
            onClick={handleAsk}
            disabled={!kbReady || !question.trim() || isAsking}
          >
            {isAsking
              ? <><span className="spinner-xs spinner-white" /> Analyzing knowledge base…</>
              : <><span>✨</span> Ask AI Vaidya</>}
          </button>

          <SampleQuestions
            questions={sampleQuestions}
            onSelect={handleSampleQuestionSelect}
            disabled={!kbReady || isAsking}
          />

          <AskStatus status={askStatus} />

          {askError && (
            <div className="answer-error"><span>⚠️</span> {askError}</div>
          )}

          {answer && (
            <div className="answer-block glass-panel" ref={answerRef}>
              <div className="answer-label"><span>✨</span> Answer</div>
              <p className="answer-text">{answer}</p>

              {sources.length > 0 && (
                <div className="sources-section">
                  <div className="sources-label">📌 Source passages</div>
                  <div className="sources-list">
                    {sources.map((s, i) => (
                      <details key={i} className="source-item">
                        <summary className="source-summary">
                          <span className="source-num">#{i + 1}</span>
                          <span className="source-file">{s.pdf}</span>
                          <span className="source-page">Page {s.page}</span>
                          <span className="source-chevron">›</span>
                        </summary>
                        <p className="source-snippet">{s.snippet}</p>
                      </details>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
