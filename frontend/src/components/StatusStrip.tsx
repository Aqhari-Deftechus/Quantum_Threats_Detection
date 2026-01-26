import { useEffect, useState } from 'react';
import { fetchStatus, fetchSystemHealth, StatusResponse, SystemHealthResponse } from '../api';
import { useNavigate } from 'react-router-dom';

const emptyStatus: StatusResponse = {
  system: 'OK',
  inference: 'CPU',
  matcher: 'degraded_cosine',
  matcher_index_status: 'degraded',
  evidence: 'VERIFY FAIL',
  cameras_live: 0,
  cameras_total: 0,
  cameras_down: 0,
  cameras_avg_fps: 0,
  cameras_avg_latency_ms: 0,
  cameras_queue_depth: 0,
  cameras_dropped_frames: 0,
  ws_status: 'RECONNECTING',
  timestamp: new Date().toISOString()
};

export default function StatusStrip({ wsConnected }: { wsConnected: boolean }) {
  const [status, setStatus] = useState<StatusResponse>(emptyStatus);
  const [systemHealth, setSystemHealth] = useState<SystemHealthResponse | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const [statusResponse, systemResponse] = await Promise.all([fetchStatus(), fetchSystemHealth()]);
        if (active) {
          setStatus(statusResponse);
          setSystemHealth(systemResponse);
        }
      } catch {
        if (active) {
          setStatus((prev) => ({ ...prev, system: 'DOWN' }));
        }
      }
    };
    load();
    const interval = setInterval(load, 2000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="flex flex-wrap gap-2 bg-[#0b1533] px-4 py-2 border-b border-white/10">
      <button className="status-pill" onClick={() => navigate('/settings')}>SYSTEM: {systemHealth?.status ?? status.system}</button>
      <button className="status-pill" onClick={() => navigate('/settings')}>INFERENCE: {status.inference}</button>
      <button className="status-pill" onClick={() => navigate('/settings')}>MATCHER: {status.matcher}</button>
      <button className="status-pill" onClick={() => navigate('/events')}>EVIDENCE: {status.evidence}</button>
      <button className="status-pill" onClick={() => navigate('/cameras')}>
        CAMERAS: LIVE {status.cameras_live} / TOTAL {status.cameras_total} (DOWN {status.cameras_down})
      </button>
      <span className="status-pill">WS: {wsConnected ? 'CONNECTED' : status.ws_status}</span>
    </div>
  );
}
