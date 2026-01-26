import { useEffect, useState } from 'react';
import { fetchStatus, fetchSystemHealth, StatusResponse, SystemHealthResponse } from '../api';

export default function Settings() {
  const [health, setHealth] = useState<SystemHealthResponse | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);

  useEffect(() => {
    fetchSystemHealth().then(setHealth).catch(() => setHealth(null));
    fetchStatus().then(setStatus).catch(() => setStatus(null));
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Settings</h1>
      <div className="deftech-card">
        <h2 className="text-lg font-semibold mb-2">System Health</h2>
        {health ? (
          <div className="space-y-1 text-sm">
            <div>Status: {health.status}</div>
            <div>FPS: {health.metrics.fps}</div>
            <div>Latency (ms): {health.metrics.latency_ms}</div>
            <div>Dropped Frames: {health.metrics.dropped_frames}</div>
          </div>
        ) : (
          <div className="text-white/60">Health data unavailable.</div>
        )}
      </div>
      <div className="deftech-card">
        <h2 className="text-lg font-semibold mb-2">Matcher</h2>
        <div className="text-sm space-y-1">
          <div>Mode: {status?.matcher ?? 'unknown'}</div>
          <div>Index Status: {status?.matcher_index_status ?? 'unknown'}</div>
        </div>
      </div>
      <div className="deftech-card">
        <h2 className="text-lg font-semibold mb-2">Policy</h2>
        <p className="text-sm">Policy enforcement and fusion signals are active in Phase 0.</p>
      </div>
    </div>
  );
}
