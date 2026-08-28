import { useState, useEffect } from 'react'
import { Upload, MessageSquare, FolderOpen, Activity, BookOpen, Circle } from 'lucide-react'
import UploadPage from './pages/Upload'
import QueryPage from './pages/Query'
import DocumentsPage from './pages/Documents'
import TracesPage from './pages/Traces'
import { getHealth } from './api'

const NAV = [
  { id: 'upload',    label: 'Upload',    Icon: Upload },
  { id: 'query',     label: 'Query',     Icon: MessageSquare },
  { id: 'documents', label: 'Documents', Icon: FolderOpen },
  { id: 'traces',    label: 'Traces',    Icon: Activity },
]

export default function App() {
  const [page, setPage] = useState('upload')
  const [health, setHealth] = useState(null)
  const [lastTrace, setLastTrace] = useState(null)

  useEffect(() => {
    getHealth().then(r => setHealth(r.data)).catch(() => setHealth({ qdrant: false, version: '?' }))
  }, [])

  return (
    <div className="flex h-screen bg-slate-900 text-white overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 shrink-0 bg-slate-900 border-r border-slate-800 flex flex-col">
        {/* Logo */}
        <div className="px-5 py-5 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <BookOpen className="text-blue-400" size={20} />
            <div>
              <p className="font-bold text-white text-sm leading-none">RAG HR App</p>
              <p className="text-slate-500 text-xs mt-0.5">Week 5 — Dynamic RAG</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV.map(({ id, label, Icon }) => (
            <button
              key={id}
              onClick={() => setPage(id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors
                ${page === id
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}
            >
              <Icon size={16} />
              {label}
            </button>
          ))}
        </nav>

        {/* Health badge */}
        <div className="px-5 py-4 border-t border-slate-800">
          {health && (
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Circle
                size={8}
                className={health.qdrant ? 'fill-emerald-400 text-emerald-400' : 'fill-red-400 text-red-400'}
              />
              <span>API v{health.version} &bull; Qdrant {health.qdrant ? 'Online' : 'Offline'}</span>
            </div>
          )}
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto p-8">
        {page === 'upload'    && <UploadPage />}
        {page === 'query'     && <QueryPage onTrace={setLastTrace} />}
        {page === 'documents' && <DocumentsPage />}
        {page === 'traces'    && <TracesPage lastResult={lastTrace} />}
      </main>
    </div>
  )
}
