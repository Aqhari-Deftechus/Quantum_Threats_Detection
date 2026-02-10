const trimTrailingSlash = (value: string) => value.replace(/\/+$/, '');

const parseBoolean = (value: string | undefined, fallback: boolean) => {
  if (value === undefined) {
    return fallback;
  }
  const normalized = value.trim().toLowerCase();
  return ['1', 'true', 'yes', 'on'].includes(normalized);
};

const getDefaultApiBaseUrl = () => {
  return 'http://127.0.0.1:8010';
};

const getDefaultWsBaseUrl = () => {
  return 'ws://127.0.0.1:8010';
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

export const ENABLE_WHEP = parseBoolean(import.meta.env.VITE_ENABLE_WHEP, false);
