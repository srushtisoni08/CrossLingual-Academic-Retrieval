import { Link } from 'react-router-dom'

export default function Navbar() {
  return (
    <header className="navbar">
      <div className="container navbar-inner">
        <Link to="/" className="brand">
          <span className="brand-mark" aria-hidden="true">अ</span>
          CrossLingual Retrieval
        </Link>
        <span className="navbar-note">English · हिन्दी · বাংলা · తెలుగు</span>
      </div>
    </header>
  )
}