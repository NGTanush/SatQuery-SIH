import React, {useState} from 'react';
import {createRoot} from 'react-dom/client';
import {MapContainer, ImageOverlay, Rectangle} from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './styles.css';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api/v1';

function App() {
  const [first, setFirst] = useState(); const [second, setSecond] = useState();
  const [query, setQuery] = useState('Describe this scene'); const [result, setResult] = useState(); const [loading, setLoading] = useState(false); const [preview, setPreview] = useState();
  async function analyze(event) {
    event.preventDefault(); if (!first) return; setLoading(true);
    const body = new FormData(); body.append('file_1', first); if (second) body.append('file_2', second);
    body.append('query', query); body.append('include_report', 'true');
    const response = await fetch(`${API}/agent`, {method: 'POST', body}); setResult(await response.json()); setLoading(false);
  }
  function downloadReport() { const raw = atob(result.report_pdf_b64); const bytes = Uint8Array.from(raw, c => c.charCodeAt(0)); const url = URL.createObjectURL(new Blob([bytes], {type:'application/pdf'})); Object.assign(document.createElement('a'), {href:url, download:'satquery-report.pdf'}).click(); }
  return <main><h1>SatQuery AI</h1><p>Remote-sensing analysis with auditable specialist routing.</p>
    <form onSubmit={analyze}><label>Primary image<input type="file" accept="image/*,.tif,.tiff" onChange={e=>{setFirst(e.target.files[0]); setPreview(URL.createObjectURL(e.target.files[0]))}}/></label>
    <label>Optional paired image<input type="file" accept="image/*,.tif,.tiff" onChange={e=>setSecond(e.target.files[0])}/></label>
    <label>Question<textarea value={query} onChange={e=>setQuery(e.target.value)}/></label><button disabled={loading}>{loading?'Analyzing…':'Analyze'}</button></form>
    {preview && <section><h2>Map viewer</h2><MapContainer className="map" center={[0,0]} zoom={1} crs={L.CRS.Simple}><ImageOverlay url={result?.overlay_b64 ? `data:image/png;base64,${result.overlay_b64}` : preview} bounds={[[-100,-100],[100,100]]}/>{result?.bounding_boxes?.map((box, index) => <Rectangle key={index} bounds={[[100-box.coordinates[0], box.coordinates[1]-100],[100-box.coordinates[2], box.coordinates[3]-100]]}/>)}</MapContainer></section>}
    {result && <section><h2>Result</h2><p>{result.answer || result.detail}</p><h3>Execution trace</h3><pre>{JSON.stringify(result.route || result.execution_trace, null, 2)}</pre>{result.report_pdf_b64 && <button onClick={downloadReport}>Download PDF report</button>}</section>}
  </main>;
}
createRoot(document.getElementById('root')).render(<App/>);
