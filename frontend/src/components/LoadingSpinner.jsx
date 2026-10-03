export default function LoadingSpinner({ label = 'Loading…' }) {
  return (
    <div className="state" role="status">
      <div className="spinner" />
      <p>{label}</p>
    </div>
  )
}