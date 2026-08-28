import { useState, useEffect } from 'react'
import { FileText, ChevronDown, ChevronRight, Table, Clock } from 'lucide-react'
import { getCollections, getDocuments, getChunks } from '../api'

export default function Documents() {
  const [collections, setCollections] = useState(['documents'])
  const [collection, setCollection] = useState('documents')
  const [docs, setDocs] = useState([])
  const [selectedDoc, setSelectedDoc] = useState(null)
  const [chunks, setChunks] = useState([])
  const [expandedChunk, setExpandedChunk] = useState(null)
  const [loadingDocs, setLoadingDocs] = useState(false)
  const [loadingChunks, setLoadingChunks] = useState(false)

  useEffect(() => {
    getCollections().then(r => setCollections([...new Set(['documents', ...r.data.map(c => c.name)])])).catch(() => {})
  }, [])

  useEffect(() => {
    setLoadingDocs(true)
    getDocuments(collection)
      .then(r => { setDocs(r.data); setSelectedDoc(null); setChunks([]) })
      .catch(() => setDocs([]))
      .finally(() => setLoadingDocs(false))
  }, [collection])

  const loadChunks = (filename) => {
    if (selectedDoc === filename) { setSelectedDoc(null); setChunks([]); return }
    setSelectedDoc(filename); setLoadingChunks(true); setExpandedChunk(null)
    getChunks(filename, collection)
      .then(r => setChunks(r.data))
      .catch(() => setChunks([]))
      .finally(() => setLoadingChunks(false))
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-white">Uploaded Documents</h1>
        <select value={collection} onChange={e => setCollection(e.target.value)}
          className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-white text-sm focus:outline-none focus:border-blue-500">
          {collections.map(c => <option key={c}>{c}</option>)}
        </select>
      </div>

      {loadingDocs && <p className="text-slate-400 text-sm">Loading...</p>}

      {!loadingDocs && docs.length === 0 && (
        <div className="bg-slate-800 rounded-xl p-10 text-center text-slate-400">
          No documents in <strong>{collection}</strong>. Upload some first.
        </div>
      )}

      {docs.length > 0 && (
        <div className="space-y-3">
          {docs.map(doc => (
            <div key={doc.filename} className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
              {/* Doc row */}
              <button
                onClick={() => loadChunks(doc.filename)}
                className="w-full flex items-center gap-4 px-5 py-4 hover:bg-slate-700/40 transition-colors text-left"
              >
                <FileText className="text-blue-400 shrink-0" size={20} />
                <div className="flex-1 min-w-0">
                  <p className="text-white font-medium truncate">{doc.filename}</p>
                  <div className="flex items-center gap-3 mt-0.5 text-xs text-slate-500">
                    <span className="uppercase">{doc.source_type}</span>
                    <span>{doc.chunk_count} chunks</span>
                    {doc.has_table && <span className="flex items-center gap-1"><Table size={10} /> has tables</span>}
                    <span className="flex items-center gap-1">
                      <Clock size={10} /> {doc.upload_timestamp?.slice(0, 19).replace('T', ' ')}
                    </span>
                  </div>
                </div>
                {selectedDoc === doc.filename
                  ? <ChevronDown size={16} className="text-slate-400 shrink-0" />
                  : <ChevronRight size={16} className="text-slate-400 shrink-0" />}
              </button>

              {/* Chunk list */}
              {selectedDoc === doc.filename && (
                <div className="border-t border-slate-700 px-5 py-3">
                  {loadingChunks ? (
                    <p className="text-slate-400 text-sm py-2">Loading chunks...</p>
                  ) : chunks.length === 0 ? (
                    <p className="text-slate-400 text-sm py-2">No chunks found.</p>
                  ) : (
                    <div className="space-y-1">
                      <p className="text-slate-500 text-xs mb-2">{chunks.length} chunks</p>
                      {chunks.map((chunk, i) => (
                        <div key={chunk.id} className="rounded-lg border border-slate-700 overflow-hidden">
                          <button
                            onClick={() => setExpandedChunk(expandedChunk === i ? null : i)}
                            className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-slate-700/40 transition-colors"
                          >
                            {expandedChunk === i ? <ChevronDown size={12} className="text-slate-500 shrink-0" /> : <ChevronRight size={12} className="text-slate-500 shrink-0" />}
                            <span className="text-xs text-slate-300 truncate flex-1">
                              Chunk {chunk.chunk_index ?? i} &mdash; {chunk.heading || '—'}
                            </span>
                          </button>
                          {expandedChunk === i && (
                            <div className="px-3 pb-3 border-t border-slate-700">
                              <pre className="text-slate-300 text-xs whitespace-pre-wrap mt-2 leading-relaxed font-sans">{chunk.text}</pre>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
