import { Link } from 'react-router-dom'
import { langName } from '../constants.js'

export default function SearchResult({ result, position }) {
  return (
    <li className="result">
      <div className="result-rank">{position}</div>
      <div className="result-body">
        <Link
          className="result-title"
          to={`/paper/${result.doc_id}?lang=${result.lang}`}
          lang={result.lang}
        >
          {result.title || 'Untitled passage'}
        </Link>
        <p className="result-snippet" lang={result.lang}>{result.snippet}</p>
        <div className="result-meta">
          <span className="badge">{langName(result.lang)}</span>
          <span>similarity {result.score.toFixed(3)}</span>
        </div>
      </div>
    </li>
  )
}