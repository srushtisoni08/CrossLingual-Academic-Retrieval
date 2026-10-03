import { useEffect, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import LanguageSelector from '../components/LanguageSelector.jsx'
import LoadingSpinner from '../components/LoadingSpinner.jsx'
import { getPaper } from '../services/api.js'

export default function PaperDetails() {
  const { docId } = useParams()
  const [params, setParams] = useSearchParams()
  const navigate = useNavigate()
  const lang = params.get('lang') ?? 'en'

  const [state, setState] = useState({ key: null, paper: null, error: null })
  const key = `${docId}|${lang}`

  useEffect(() => {
    const controller = new AbortController()
    getPaper(docId, lang, controller.signal)
      .then((paper) => setState({ key, paper, error: null }))
      .catch((err) => {
        if (!controller.signal.aborted) setState({ key, paper: null, error: err.message })
      })
    return () => controller.abort()
  }, [key, docId, lang])

  const loading = state.key !== key
  const goBack = () => (window.history.length > 1 ? navigate(-1) : navigate('/'))

  return (
    <article className="paper">
      <button type="button" className="link-btn" onClick={goBack}>← Back to results</button>
      <LanguageSelector
        label="Read in"
        value={lang}
        onChange={(l) => setParams({ lang: l }, { replace: true })}
      />
      {loading && <LoadingSpinner />}
      {!loading && state.error && <div className="state error">{state.error}</div>}
      {!loading && state.paper && (
        <>
          <h1 lang={lang}>{state.paper.title || 'Untitled passage'}</h1>
          {state.paper.text.split('\n').filter(Boolean).map((para, i) => (
            <p key={i} lang={lang}>{para}</p>
          ))}
          <p className="muted">
            This is the same passage in every language — switch above to compare translations.
          </p>
        </>
      )}
    </article>
  )
}