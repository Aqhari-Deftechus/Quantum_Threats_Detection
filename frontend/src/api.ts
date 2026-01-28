export const API_BASE = 'http://localhost:8000/api';

export type StatusResponse = {
  system: string;
  inference: string;
  matcher: string;
  matcher_index_status: string;
  evidence: string;
  cameras_live: number;
  cameras_total: number;
  cameras_down: number;
  cameras_avg_fps: number;
  cameras_avg_latency_ms: number;
  cameras_queue_depth: number;
  cameras_dropped_frames: number;
  ws_status: string;
  timestamp: string;
};

export type SystemHealthResponse = {
  status: string;
  metrics: Record<string, number>;
  timestamp: string;
};

export type Camera = {
  id: number;
  name: string;
  source_type: string;
  source_redacted: string;
  enabled: boolean;
  decoder_mode: string;
  camera_status: string;
};

export type CameraHealth = {
  camera_id: number;
  status: string;
  last_seen: string | null;
  fps: number;
  dropped_frames: number;
  queue_depth: number;
  latency_ms: number;
};

export type Identity = {
  id: number;
  name: string;
  notes: string;
  created_at: string;
  updated_at: string;
  embedding_count: number;
};

export async function fetchStatus(): Promise<StatusResponse> {
  const response = await fetch(`${API_BASE}/status`);
  return response.json();
}

export async function fetchSystemHealth(): Promise<SystemHealthResponse> {
  const response = await fetch(`${API_BASE}/system/health`);
  return response.json();
}

export async function fetchCameras(): Promise<Camera[]> {
  const response = await fetch(`${API_BASE}/cameras`);
  return response.json();
}

export async function fetchCameraHealth(cameraId: number): Promise<CameraHealth> {
  const response = await fetch(`${API_BASE}/cameras/${cameraId}/health`);
  return response.json();
}

export async function createCamera(payload: {
  name: string;
  source: string;
  source_type: string;
  enabled: boolean;
  decoder_mode: string;
}): Promise<Camera> {
  const response = await fetch(`${API_BASE}/cameras`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  return response.json();
}

export async function fetchIdentities(): Promise<Identity[]> {
  const response = await fetch(`${API_BASE}/identities`);
  return response.json();
}

export async function createIdentity(payload: { name: string; notes: string }): Promise<Identity> {
  const response = await fetch(`${API_BASE}/identities`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  return response.json();
}

export async function enrollIdentityEmbedding(identityId: number): Promise<Identity> {
  const response = await fetch(`${API_BASE}/identities/${identityId}/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({})
  });
  return response.json();
}
