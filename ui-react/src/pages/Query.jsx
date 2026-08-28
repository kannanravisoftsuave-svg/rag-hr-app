import { useState, useEffect } from 'react'
import { Send, ChevronDown, ChevronRight, ExternalLink, AlertTriangle } from 'lucide-react'
import { queryDocuments, getCollections, getDocuments } from '../api'

const LANGFUSE_HOST = 'https://cloud.langfuse.com'

export default function Query({ onTrace }) {
  const [question, setQuestion] = useState('')
  const [collection, setCollection] = useState('documents')
  const [collections, setCollections] = useState(['documents'])
  const [docs, setDocs] = useState([])
  const [filterFile, setFilterFile] = useState('')
  const [topK, setTopK] = useState(8)
  const [threshold, setThreshold] = useState(0.45)
  const [hybrid, setHybrid] = useState(true)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [expandedChunk, setExpandedChunk] = useState(null)

  useEffect(() => {
    getCollections().then(r => setCollections([...new Set(['documents', ...r.data.map(c => c.name)])])).catch(() => {})
  }, [])

  useEffect(() => {
    getDocuments(collection).then(r => setDocs(r.data)).catch(() => setDocs([]))
    setFilterFile('')
  }, [collection])

  const handleAsk = async (e) => {
    e.preventDefault()
    if (!question.trim()) return
    setLoading(true); setResult(null); setError(null)
    try {
      const payload = {
        question,
        collection,
        top_k: topK,
        use_hybrid: hybrid,
        confidence_threshold: threshold,
        filters: filterFile ? { source_file: filterFile } : null,
      }
      const r = await queryDocuments(payload)
      setResult(r.data)
      if (onTrace) onTrace(r.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const confColor = (c) => c >= 0.8 ? 'text-emerald-400' : c >= threshold ? 'text-yellow-400' : 'text-red-400'
  const confDot = (c) => c >= 0.8 ? 'bg-emerald-400' : c >= threshold ? 'bg-yellow-400' : 'bg-red-400'

  return (
    <div className="flex gap-6 h-full">
      {/* Settings sidebar */}
      <aside className="w-60 shrink-0 space-y-4">
        <h2 className="text-slate-400 text-xs font-semibold uppercase tracking-wider">Settings</h2>

        <div>
          <label className="block text-xs text-slate-500 mb-1">Collection</label>
          <select value={collection} onChange={e => setCollection(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-white text-sm focus:outline-none focus:border-blue-500">
            {collections.map(c => <option key={c}>{c}</option>)}
          </select>
        </div>

        <div>
          <label className="block text-xs text-slate-500 mb-1">Filter by document</label>
          <select value={filterFile} onChange={e => setFilterFile(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-white text-sm focus:outline-none focus:border-blue-500">
            <option value="">-- all --</option>
            {docs.map(d => <option key={d.filename}>{d.filename}</option>)}
          </select>
        </div>

        <div>
          <label className="block text-xs text-slate-500 mb-1">Top K: {topK}</label>
          <input type="range" min={1} max={20} value={topK} onChange={e => setTopK(+e.target.value)}
            className="w-full accent-blue-500" />
        </div>

        <div>
          <label className="block text-xs text-slate-500 mb-1">Threshold: {threshold.toFixed(2)}</label>
          <input type="range" min={0} max={1} step={0.05} value={threshold} onChange={e => setThreshold(+e.target.value)}
            className="w-full accent-blue-500" />
        </div>

        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={hybrid} onChange={e => setHybrid(e.target.checked)}
            className="accent-blue-500 w-4 h-4" />
          <span className="text-sm text-slate-300">Hybrid search</span>
        </label>
      </aside>

      {/* Main area */}
      <div className="flex-1 flex flex-col gap-4">
        <h1 className="text-2xl font-bold text-white">Ask a Question</h1>

        <form onSubmit={handleAsk} className="flex gap-2">
          <textarea
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleAsk(e) } }}
            rows={3}
            placeholder="e.g. Who is the CEO of ACME?"
            className="flex-1 bg-slate-800 border border-slate-600 rounded-xl px-4 py-3 text-white text-sm resize-none focus:outline-none focus:border-blue-500 placeholder-slate-500"
          />
          <button type="submit" disabled={loading || !question.trim()}
            className="self-end bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white px-5 py-3 rounded-xl transition-colors">
            <Send size={18} />
          </button>
        </form>

        {loading && (
          <div className="flex items-center gap-3 text-slate-400">
            <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            Retrieving and generating...
          </div>
        )}

        {error && (
          <div className="bg-red-950 border border-red-700 rounded-xl p-4 flex items-start gap-3">
            <AlertTriangle className="text-red-400 shrink-0" size={18} />
            <p className="text-red-300 text-sm">{error}</p>
          </div>
        )}

        {result && (
          <div className="space-y-4">
            {/* Confidence bar */}
            <div className="flex items-center gap-3 bg-slate-800 rounded-xl px-4 py-2.5">
              <div className={`w-2 h-2 rounded-full ${confDot(result.confidence)}`} />
              <span className={`font-semibold text-sm ${confColor(result.confidence)}`}>
                Confidence {(result.confidence * 100).toFixed(1)}%
              </span>
              <span className="text-slate-500 text-xs ml-auto">
                {result.collection} &bull; {result.model?.split('/').pop()}
              </span>
            </div>

            {/* Answer */}
            {result.answer ? (
              <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
                <h2 className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-3">Answer</h2>
                <div className="text-white text-sm leading-relaxed whitespace-pre-wrap">{result.answer}</div>
              </div>
            ) : (
              <div className="bg-amber-950 border border-amber-700 rounded-xl p-4">
                <p className="text-amber-300 text-sm">{result.message || 'No answer generated.'}</p>
                <p className="text-amber-500 text-xs mt-1">Reason: {result.reason}</p>
              </div>
            )}

            {/* Chunks */}
            {result.chunks?.length > 0 && (
              <div>
                <h2 className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
                  Retrieved Chunks ({result.chunks.length})
                </h2>
                <div className="space-y-2">
                  {result.chunks.map((chunk, i) => (
                    <div key={i} className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
                      <button
                        onClick={() => setExpandedChunk(expandedChunk === i ? null : i)}
                        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-slate-700/50 transition-colors"
                      >
                        {expandedChunk === i ? <ChevronDown size={14} className="text-slate-400 shrink-0" /> : <ChevronRight size={14} className="text-slate-400 shrink-0" />}
                        <span className="text-xs text-slate-300 flex-1 truncate">
                          #{i+1} &mdash; <span className="text-blue-400">{chunk.source_file}</span> &rsaquo; {chunk.heading}
                        </span>
                        <span className="text-xs font-mono text-slate-400 shrink-0">
                          {(chunk.similarity * 100).toFixed(1)}%
                        </span>
                      </button>
                      {expandedChunk === i && (
                        <div className="px-4 pb-4 border-t border-slate-700">
                          <pre className="text-slate-300 text-xs whitespace-pre-wrap mt-3 leading-relaxed font-sans">{chunk.text}</pre>
                          <div className="flex gap-4 mt-3 text-xs text-slate-500">
                            <span>Type: {chunk.source_type}</span>
                            <span>Similarity: {chunk.similarity.toFixed(4)}</span>
                            {chunk.rrf_score && <span>RRF: {chunk.rrf_score.toFixed(4)}</span>}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Trace link */}
            {result.trace_url && (
              <a href={result.trace_url} target="_blank" rel="noreferrer"
                className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 text-sm">
                <ExternalLink size={14} /> View full trace in Langfuse
              </a>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
