import { langName } from '../constants.js'
import SearchResult from './SearchResult.jsx'

// The dataset contains some identical passages under different ids;
// show each distinct passage once.
function dedupe(results) {
  const seen = new Set()
  return results.filter((r) => {
    const key = `${r.title}|${r.snippet}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

export default function SearchResults({ data }) {
  const results = dedupe(data.results)
  const cross = data.query_lang !== data.corpus_lang

  return (
    <section>
      <div className="results-head">
        <strong>{results.length} results</strong>
        <span className="muted">
          {cross
            ? `${langName(data.query_lang)} query → ${langName(data.corpus_lang)} passages`
            : `${langName(data.corpus_lang)} passages`}
          {' · '}{data.mode} · {data.took_ms} ms
        </span>
      </div>
      {results.length === 0 ? (
        <div className="state"><p>No results. Try rephrasing your question.</p></div>
      ) : (
        <ol className="results">
          {results.map((r, i) => (
            <SearchResult key={r.doc_id} result={r} position={i + 1} />
          ))}
        </ol>
      )}
    </section>
  )
}