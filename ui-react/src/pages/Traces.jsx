import { ExternalLink } from 'lucide-react'

const LANGFUSE_HOST = 'https://cloud.langfuse.com'

const SPANS = [
  ['retrieval', 'Query text, hit count, top similarity scores'],
  ['reranking', 'Candidate count, top cross-encoder score'],
  ['confidence_gate', 'Best similarity vs threshold, pass/fail'],
  ['llm_generation', 'Full prompt, model response'],
  ['text_extraction', 'File format, char count, table detection, warnings'],
  ['chunking', 'Strategy used, chunk count, avg size'],
  ['embedding', 'Model name, chunk count'],
  ['vector_store', 'Collection, number stored'],
]

export default function Traces({ lastResult }) {
  return (
    <div className="max-w-3xl">
      <h1 className="text-2xl font-bold text-white mb-1">Langfuse Traces</h1>
      <p className="text-slate-400 text-sm mb-6">Every query and upload creates a trace with nested spans.</p>

      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700">
              <th className="text-left px-4 py-3 text-slate-400 font-medium">Span</th>
              <th className="text-left px-4 py-3 text-slate-400 font-medium">What it records</th>
            </tr>
          </thead>
          <tbody>
            {SPANS.map(([span, desc], i) => (
              <tr key={span} className={i % 2 === 0 ? 'bg-slate-800' : 'bg-slate-800/50'}>
                <td className="px-4 py-2.5">
                  <code className="text-blue-400 text-xs bg-blue-950 px-1.5 py-0.5 rounded">{span}</code>
                </td>
                <td className="px-4 py-2.5 text-slate-300 text-xs">{desc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <a
        href={LANGFUSE_HOST}
        target="_blank"
        rel="noreferrer"
        className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold px-5 py-2.5 rounded-lg transition-colors mb-8"
      >
        <ExternalLink size={16} /> Open Langfuse Dashboard
      </a>

      <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 mb-6">
        <h2 className="text-white font-semibold mb-3">Week 5 Module 3 — Error Analysis</h2>
        <ol className="space-y-2 text-slate-300 text-sm list-decimal list-inside">
          <li>Run <strong>20 real queries</strong> from the Query page (mix of on-topic, off-topic, ambiguous, multi-hop)</li>
          <li>Open each trace in Langfuse and read the full span tree</li>
          <li>Write one honest sentence per failure noting <strong>what went wrong</strong></li>
          <li>Group the notes into ~5 named problem types</li>
          <li>Rank by frequency × severity → your ranked taxonomy deliverable</li>
        </ol>
      </div>

      {/* Last query trace */}
      {lastResult?.trace_id && (
        <div className="bg-slate-800 rounded-xl p-5 border border-slate-700">
          <h2 className="text-white font-semibold mb-3">Last Query Trace</h2>
          <div className="space-y-2 text-sm">
            <div className="flex gap-2">
              <span className="text-slate-500 w-28 shrink-0">Question</span>
              <span className="text-slate-200">{lastResult.question}</span>
            </div>
            <div className="flex gap-2">
              <span className="text-slate-500 w-28 shrink-0">Confidence</span>
              <span className="text-slate-200">{(lastResult.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="flex gap-2">
              <span className="text-slate-500 w-28 shrink-0">Reason</span>
              <span className="text-slate-200">{lastResult.reason ?? 'answered'}</span>
            </div>
            <div className="flex gap-2">
              <span className="text-slate-500 w-28 shrink-0">Trace ID</span>
              <code className="text-blue-400 text-xs">{lastResult.trace_id}</code>
            </div>
          </div>
          {lastResult.trace_url && (
            <a href={lastResult.trace_url} target="_blank" rel="noreferrer"
              className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 text-sm mt-4">
              <ExternalLink size={14} /> View this trace in Langfuse
            </a>
          )}
        </div>
      )}
    </div>
  )
}
