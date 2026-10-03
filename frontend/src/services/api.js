const BASE = '/api'

async function request(path, options) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, options)
  } catch (err) {
    if (err.name === 'AbortError') throw err
    throw new Error('Cannot reach the server. Is the backend running on port 8000?', { cause: err })
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      /* response had no JSON body */
    }
    throw new Error(detail)
  }
  return res.json()
}

export function searchPapers({ query, corpusLang = 'en', mode = 'dense', topK = 10, signal }) {
  return request('/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, corpus_lang: corpusLang, mode, top_k: topK }),
    signal,
  })
}

export function getPaper(docId, lang = 'en', signal) {
  return request(`/papers/${encodeURIComponent(docId)}?lang=${lang}`, { signal })
}