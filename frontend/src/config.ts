const trimTrailingSlash = (value: string) => value.replace(/\/+$/, '');

const getDefaultApiBaseUrl = () => {
  if (import.meta.env.DEV) {
    return 'http://127.0.0.1:8010';
  }
  return window.location.origin;
};

const getDefaultWsBaseUrl = () => {
  if (import.meta.env.DEV) {
    return 'ws://127.0.0.1:8010';
  }
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${protocol}://${window.location.host}`;
};

export const API_BASE_URL = trimTrailingSlash(
  import.meta.env.VITE_API_BASE_URL ?? getDefaultApiBaseUrl()
);

export const WS_BASE_URL = trimTrailingSlash(
  import.meta.env.VITE_WS_BASE_URL ?? getDefaultWsBaseUrl()
);

export const MEDIAMTX_WHEP_BASE_URL = trimTrailingSlash(
  import.meta.env.VITE_MEDIAMTX_WHEP_BASE_URL ?? 'http://127.0.0.1:8889'
);

export const MEDIAMTX_WHEP_PATH_TEMPLATE =
  import.meta.env.VITE_MEDIAMTX_WHEP_PATH_TEMPLATE ?? 'camera-{camera_id}';
