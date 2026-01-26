import { useEffect, useState } from 'react';
import { createCamera, fetchCameras, Camera } from '../api';

const emptyForm = {
  name: '',
  source: '',
  source_type: 'WEBCAM',
  enabled: true,
  decoder_mode: 'opencv'
};

export default function Cameras() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [form, setForm] = useState(emptyForm);

  const load = async () => {
    try {
      const data = await fetchCameras();
      setCameras(data);
    } catch {
      setCameras([]);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    await createCamera(form);
    setForm(emptyForm);
    load();
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Cameras</h1>
      <div className="deftech-card">
        <form className="grid grid-cols-1 md:grid-cols-2 gap-4" onSubmit={handleSubmit}>
          <input
            className="bg-[#122454] border border-white/20 rounded px-2 py-1"
            placeholder="Camera Name"
            value={form.name}
            onChange={(event) => setForm({ ...form, name: event.target.value })}
            required
          />
          <input
            className="bg-[#122454] border border-white/20 rounded px-2 py-1"
            placeholder="Source (0 or rtsp://...)"
            value={form.source}
            onChange={(event) => setForm({ ...form, source: event.target.value })}
            required
          />
          <select
            className="bg-[#122454] border border-white/20 rounded px-2 py-1"
            value={form.source_type}
            onChange={(event) => setForm({ ...form, source_type: event.target.value })}
          >
            <option value="WEBCAM">WEBCAM</option>
            <option value="RTSP">RTSP</option>
            <option value="VIDEO_FILE">VIDEO_FILE</option>
          </select>
          <select
            className="bg-[#122454] border border-white/20 rounded px-2 py-1"
            value={form.decoder_mode}
            onChange={(event) => setForm({ ...form, decoder_mode: event.target.value })}
          >
            <option value="opencv">opencv</option>
            <option value="ffmpeg">ffmpeg</option>
            <option value="none">none</option>
          </select>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={form.enabled}
              onChange={(event) => setForm({ ...form, enabled: event.target.checked })}
            />
            Enabled
          </label>
          <button className="bg-green px-3 py-2 rounded" type="submit">
            Add Camera
          </button>
        </form>
      </div>

      <div className="deftech-card">
        <h2 className="text-lg font-semibold mb-2">Registered Cameras</h2>
        <div className="space-y-2">
          {cameras.length === 0 && <div className="text-white/60">No cameras registered.</div>}
          {cameras.map((camera) => (
            <div key={camera.id} className="flex items-center justify-between border-b border-white/10 py-2">
              <div>
                <div className="font-semibold">{camera.name}</div>
                <div className="text-xs text-white/70">
                  {camera.source_type} | {camera.source_redacted}
                </div>
              </div>
              <div className="text-xs">Status: {camera.camera_status}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
