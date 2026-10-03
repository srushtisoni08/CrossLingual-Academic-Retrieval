import { LANGUAGES } from '../constants.js'

export default function LanguageSelector({ label, value, onChange }) {
  return (
    <div className="lang-select" role="radiogroup" aria-label={label}>
      <span className="lang-select-label">{label}</span>
      {LANGUAGES.map((l) => (
        <button
          key={l.code}
          type="button"
          role="radio"
          aria-checked={value === l.code}
          className={`chip ${value === l.code ? 'chip-active' : ''}`}
          onClick={() => onChange(l.code)}
          lang={l.code}
        >
          {l.native}
        </button>
      ))}
    </div>
  )
}