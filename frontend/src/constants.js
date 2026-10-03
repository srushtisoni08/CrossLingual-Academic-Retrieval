export const LANGUAGES = [
  { code: 'en', name: 'English', native: 'English' },
  { code: 'hi', name: 'Hindi', native: 'हिन्दी' },
  { code: 'bn', name: 'Bengali', native: 'বাংলা' },
  { code: 'te', name: 'Telugu', native: 'తెలుగు' },
]

export const langName = (code) =>
  LANGUAGES.find((l) => l.code === code)?.name ?? code

// Build a /search URL. Only non-default values go in the query string.
export function searchUrl({ query, corpusLang = 'en', mode, topK }) {
  const p = new URLSearchParams({ q: query, corpus_lang: corpusLang })
  if (mode && mode !== 'dense') p.set('mode', mode)
  if (topK && topK !== 10) p.set('top_k', String(topK))
  return `/search?${p}`
}