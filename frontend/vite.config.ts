import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFile } from 'node:fs/promises';

const root = resolve(dirname(fileURLToPath(import.meta.url)));
const indexPath = resolve(root, 'index.html');

const spaIndexFallback = () => ({
  name: 'spa-index-fallback',
  apply: 'serve',
  enforce: 'pre',
  configureServer(server) {
    server.middlewares.use(async (req, res, next) => {
      if (req.method && !['GET', 'HEAD'].includes(req.method)) {
        next();
        return;
      }

      const url = req.url?.split('?')[0] ?? '/';
      const isAssetRequest =
        url.startsWith('/@') ||
        url.startsWith('/src') ||
        url.startsWith('/node_modules') ||
        url.startsWith('/__vite') ||
        url.includes('.') ||
        url.startsWith('/api');

      if (isAssetRequest) {
        next();
        return;
      }

      try {
        const html = await readFile(indexPath, 'utf-8');
        const transformed = await server.transformIndexHtml(url, html);
        res.statusCode = 200;
        res.setHeader('Content-Type', 'text/html');
        res.end(transformed);
      } catch (error) {
        next(error);
      }
    });
  }
});

export default defineConfig({
  root,
  base: '/',
  plugins: [react(), spaIndexFallback()],
  server: {
    host: true,
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8010',
        changeOrigin: true
      },
      '/ws': {
        target: 'ws://127.0.0.1:8010',
        ws: true,
        changeOrigin: true
      }
    }
  },
  appType: 'spa'
});
