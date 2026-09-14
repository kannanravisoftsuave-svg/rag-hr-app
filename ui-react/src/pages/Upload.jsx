import { useState, useEffect } from 'react'
import { UploadCloud, CheckCircle, AlertCircle, FileText, ExternalLink } from 'lucide-react'
import { uploadDocument, getCollections } from '../api'

export default function Upload() {
  const [file, setFile] = useState(null)
  const [collection, setCollection] = useState('documents')
  const [strategy, setStrategy] = useState('auto')
  const [collections, setCollections] = useState(['documents'])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [drag, setDrag] = useState(false)

  useEffect(() => {
    getCollections()
      .then(r => setCollections([...new Set(['documents', ...r.data.map(c => c.name)])]))
      .catch(() => {})
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) return
    setLoading(true); setResult(null); setError(null)
    try {
      const r = await uploadDocument(file, collection, strategy)
      setResult(r.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const onDrop = (e) => {
    e.preventDefault(); setDrag(false)
    const f = e.dataTransfer.files[0]
    if (f) setFile(f)
  }

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold text-white mb-1">Upload Documents</h1>
      <p className="text-slate-400 text-sm mb-6">
        Upload a PDF, DOCX, or Markdown file. The system will extract, chunk, embed and store it automatically.
      </p>

      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Drop zone */}
        <div
          onDragOver={e => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
          onClick={() => document.getElementById('file-input').click()}
          className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors
            ${drag ? 'border-blue-500 bg-blue-950' : 'border-slate-600 hover:border-slate-400 bg-slate-800/50'}`}
        >
          <input
            id="file-input"
            type="file"
            accept=".pdf,.docx,.md"
            className="hidden"
            onChange={e => setFile(e.target.files[0])}
          />
          <UploadCloud className="mx-auto mb-3 text-slate-400" size={40} />
          {file ? (
            <div>
              <p className="text-white font-medium flex items-center justify-center gap-2">
                <FileText size={16} className="text-blue-400" /> {file.name}
              </p>
              <p className="text-slate-400 text-sm mt-1">{(file.size / 1024).toFixed(1)} KB</p>
            </div>
          ) : (
            <div>
              <p className="text-slate-300">Drag & drop or click to choose</p>
              <p className="text-slate-500 text-sm mt-1">PDF, DOCX, Markdown — max 10 MB</p>
            </div>
          )}
        </div>

        {/* Options row */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-slate-400 mb-1">Target collection</label>
            <select
              value={collection}
              onChange={e => setCollection(e.target.value)}
              className="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
            >
              {collections.map(c => <option key={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">Chunking strategy</label>
            <select
              value={strategy}
              onChange={e => setStrategy(e.target.value)}
              className="w-full bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
            >
              {['auto', 'heading', 'page', 'sliding_window'].map(s => <option key={s}>{s}</option>)}
            </select>
          </div>
        </div>

        <button
          type="submit"
          disabled={!file || loading}
          className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-semibold py-2.5 rounded-lg transition-colors"
        >
          {loading ? 'Ingesting...' : 'Upload & Ingest'}
        </button>
      </form>

      {/* Result */}
      {result && (
        <div className="mt-6 bg-slate-800 rounded-xl p-5 border border-emerald-700">
          <div className="flex items-center gap-2 text-emerald-400 font-semibold mb-4">
            <CheckCircle size={18} /> Ingested {result.filename} into <span className="text-white">{result.collection}</span>
          </div>
          <div className="grid grid-cols-3 gap-4 mb-4">
            {[['Chunks', result.chunk_count], ['Characters', result.char_count?.toLocaleString()], ['Pages', result.total_pages ?? '-']].map(([k, v]) => (
              <div key={k} className="bg-slate-700 rounded-lg p-3 text-center">
                <p className="text-slate-400 text-xs">{k}</p>
                <p className="text-white text-xl font-bold">{v}</p>
              </div>
            ))}
          </div>
          {result.warnings?.length > 0 && result.warnings.map((w, i) => (
            <p key={i} className="text-amber-400 text-sm">⚠ {w}</p>
          ))}
          {result.trace_id && (
            <a
              href={`https://cloud.langfuse.com/trace/${result.trace_id}`}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-300 text-sm mt-2"
            >
              <ExternalLink size={14} /> View ingestion trace in Langfuse
            </a>
          )}
        </div>
      )}

      {error && (
        <div className="mt-4 bg-red-950 border border-red-700 rounded-xl p-4 flex items-start gap-3">
          <AlertCircle className="text-red-400 shrink-0 mt-0.5" size={18} />
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}
    </div>
  )
}
