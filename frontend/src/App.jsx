import { Link, Route, Routes } from 'react-router-dom'
import Navbar from './components/Navbar.jsx'
import Home from './pages/Home.jsx'
import PaperDetails from './pages/PaperDetails.jsx'
import Search from './pages/Search.jsx'

function NotFound() {
  return (
    <div className="state">
      <h1>Page not found</h1>
      <Link to="/">Back to search</Link>
    </div>
  )
}

export default function App() {
  return (
    <>
      <Navbar />
      <main className="container">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/search" element={<Search />} />
          <Route path="/paper/:docId" element={<PaperDetails />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </>
  )
}