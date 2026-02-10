import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  // Production mode by default
  mode: 'production',
  plugins: [
    react({
      // Use automatic JSX runtime (smaller bundles)
      jsxRuntime: 'automatic',
      // Fast Refresh for hot updates
      fastRefresh: true,
      // OPTIMIZATION: Babel plugins for production optimization
      babel: {
        plugins: [
          // Remove prop-types in production for smaller bundle
          process.env.NODE_ENV === 'production' && ['transform-react-remove-prop-types', { removeImport: true }],
        ].filter(Boolean),
      },
    }),
  ],
  server: {
    port: 3000,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
    // Pre-bundle for faster startup
    warmup: {
      clientFiles: ['./src/App.tsx', './src/main.tsx'],
    },
  },
  build: {
    // Enable code splitting for better performance
    rollupOptions: {
      output: {
        manualChunks: {
          // Core React - loaded first (smallest possible)
          'react-core': ['react', 'react-dom'],
          // Routing - loaded on navigation
          'router': ['react-router-dom'],
          // UI essentials - critical path
          'ui-core': ['zustand', 'clsx', 'tailwind-merge'],
          // Icons - separate chunk for tree-shaking
          'icons': ['lucide-react'],
          // Heavy markdown rendering - lazy loaded
          'markdown': ['react-markdown', 'remark-gfm', 'remark-math', 'rehype-katex'],
          // Syntax highlighting - lazy loaded per code block
          'syntax': ['react-syntax-highlighter'],
          // WebGL - only for landing page, lazy loaded
          'webgl': ['ogl'],
          // Math rendering - loaded with markdown
          'math': ['katex'],
        },
        // OPTIMIZATION: Use hashed filenames for better caching
        chunkFileNames: 'assets/[name]-[hash].js',
        entryFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]',
      },
    },
    // Target modern browsers for smaller bundles (ES2020 = ~30% smaller)
    target: 'es2020',
    // Minify with esbuild for faster builds
    minify: 'esbuild',
    // Enable source maps only in development
    sourcemap: false,
    // Chunk size warnings
    chunkSizeWarningLimit: 500,
    // CSS code splitting
    cssCodeSplit: true,
    // Asset inlining threshold (4kb)
    assetsInlineLimit: 4096,
    // OPTIMIZATION: Enable module preload polyfill for older browsers
    modulePreload: {
      polyfill: true,
    },
    // OPTIMIZATION: Enable CSS minification
    cssMinify: true,
    // OPTIMIZATION: Report compressed size for better insights
    reportCompressedSize: true,
  },
  // Optimize dependencies
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      'zustand',
      'zustand/shallow',
      'zustand/middleware',
      'lucide-react',
      'clsx',
      'tailwind-merge',
    ],
    // Exclude heavy deps from pre-bundling (loaded lazily)
    exclude: ['react-syntax-highlighter', 'ogl'],
    // OPTIMIZATION: Faster dependency discovery
    esbuildOptions: {
      target: 'es2020',
    },
  },
  // Reduce bundle by excluding unused exports
  esbuild: {
    // Drop console in production
    drop: process.env.NODE_ENV === 'production' ? ['console', 'debugger'] : [],
    // Minimize identifiers
    minifyIdentifiers: true,
    minifySyntax: true,
    // OPTIMIZATION: Better tree shaking
    treeShaking: true,
    // Faster builds
    legalComments: 'none',
  },
  // OPTIMIZATION: Preview server caching
  preview: {
    headers: {
      'Cache-Control': 'public, max-age=31536000, immutable',
    },
  },
  // OPTIMIZATION: Resolve aliases for faster module resolution
  resolve: {
    alias: {
      '@': '/src',
    },
  },
})
