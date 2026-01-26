/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"] ,
  theme: {
    extend: {
      colors: {
        navy: '#0F1E45',
        red: '#E30613',
        green: '#4CAF50',
        amber: '#FFC107'
      },
      fontFamily: {
        rajdhani: ['Rajdhani', 'sans-serif'],
        mono: ['Roboto Mono', 'monospace']
      }
    }
  },
  plugins: []
};
