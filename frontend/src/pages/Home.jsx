import { Link, useNavigate } from 'react-router-dom'
import SearchBar from '../components/SearchBar.jsx'
import { searchUrl } from '../constants.js'

// The same question in four languages (all exist in the dataset)
const EXAMPLES = [
  { lang: 'en', q: 'When was quantum field theory developed?' },
  { lang: 'hi', q: 'क्वांटम क्षेत्र सिद्धांत का विकास कब हुआ?' },
  { lang: 'bn', q: 'কোয়ান্টাম ক্ষেত্র তত্ত্ব কখন উদ্ভাবিত হয়েছিল?' },
  { lang: 'te', q: 'క్వాంటం క్షేత్ర సిద్ధాంతం ఎప్పుడు అభివృద్ధి చేయబడింది?' },
]

export default function Home() {
  const navigate = useNavigate()

  return (
    <section className="hero">
      <h1>Search across languages</h1>
      <p className="lede">
        Ask in English, Hindi, Bengali or Telugu — and find the answer in any of them.
      </p>
      <SearchBar
        autoFocus
        onSearch={({ query, corpusLang }) => navigate(searchUrl({ query, corpusLang }))}
      />
      <div className="examples">
        <p className="muted">One question, four languages — each searches the English passages:</p>
        <div className="example-list">
          {EXAMPLES.map((ex) => (
            <Link
              key={ex.lang}
              className="example"
              lang={ex.lang}
              to={searchUrl({ query: ex.q, corpusLang: 'en' })}
            >
              {ex.q}
            </Link>
          ))}
        </div>
      </div>
    </section>
  )
}