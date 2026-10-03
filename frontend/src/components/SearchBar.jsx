import { useState } from 'react'
import LanguageSelector from './LanguageSelector.jsx'

export default function SearchBar({
  initialQuery = '',
  initialCorpusLang = 'en',
  onSearch,
  autoFocus = false,
}) {
  const [query, setQuery] = useState(initialQuery)
  const [corpusLang, setCorpusLang] = useState(initialCorpusLang)

  function submit(e) {
    e.preventDefault()
    const q = query.trim()
    if (q) onSearch({ query: q, corpusLang })
  }

  return (
    <form className="searchbar" onSubmit={submit}>
      <div className="searchbar-row">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask in any language…"
          aria-label="Search query"
          autoFocus={autoFocus}
          maxLength={500}
        />
        <button type="submit" className="btn-primary">Search</button>
      </div>
      <LanguageSelector label="Search in" value={corpusLang} onChange={setCorpusLang} />
    </form>
  )
}