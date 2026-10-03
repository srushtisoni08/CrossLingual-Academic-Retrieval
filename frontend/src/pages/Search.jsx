import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import LoadingSpinner from '../components/LoadingSpinner.jsx'
import SearchBar from '../components/SearchBar.jsx'
import SearchResults from '../components/SearchResults.jsx'
import { searchUrl } from '../constants.js'
import { searchPapers } from '../services/api.js'

export default function Search() {
  const [params] = useSearchParams()
  const navigate = useNavigate()

  const q = params.get('q') ?? ''
  const corpusLang = params.get('corpus_lang') ?? 'en'
  const mode = params.get('mode') ?? 'dense'
  const topK = Number(params.get('top_k')) || 10

  // The result is tagged with the request it belongs to, so "loading" is
  // simply "the stored result is for an older request".
  const [state, setState] = useState({ key: null, data: null, error: null })
  const key = [q, corpusLang, mode, topK].join('|')

  useEffect(() => {
    if (!q) return
    const controller = new AbortController()
    searchPapers({ query: q, corpusLang, mode, topK, signal: controller.signal })
      .then((data) => setState({ key, data, error: null }))
      .catch((err) => {
        if (!controller.signal.aborted) setState({ key, data: null, error: err.message })
      })
    return () => controller.abort()
  }, [key, q, corpusLang, mode, topK])

  const loading = Boolean(q) && state.key !== key

  return (
    <>
      <SearchBar
        key={`${q}|${corpusLang}`}
        initialQuery={q}
        initialCorpusLang={corpusLang}
        onSearch={({ query, corpusLang: lang }) =>
          navigate(searchUrl({ query, corpusLang: lang, mode, topK }))
        }
      />
      {!q && <div className="state"><p>Type a question to start searching.</p></div>}
      {loading && <LoadingSpinner label="Searching…" />}
      {!loading && state.error && <div className="state error">{state.error}</div>}
      {!loading && state.data && !state.error && <SearchResults data={state.data} />}
    </>
  )
}