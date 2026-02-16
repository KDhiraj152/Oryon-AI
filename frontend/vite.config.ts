import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [
    react({
      // Use automatic JSX runtime (smaller bundles)
      jsxRuntime: 'automatic',
      // Fast Refresh for hot updates
      fastRefresh: true,
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
    // Pre-transform files for faster page loads in dev
    warmup: {
      clientFiles: [
        './src/main.tsx',
        './src/App.tsx',
        './src/store/index.ts',
        // Chat page and its full component tree
        './src/pages/Chat.tsx',
        './src/components/chat/ChatMessage.tsx',
        './src/components/chat/ChatInput.tsx',
        './src/components/chat/ChatIndicators.tsx',
        './src/components/chat/EmptyState.tsx',
        './src/components/chat/MarkdownRenderer.tsx',
        './src/components/chat/Sidebar.tsx',
        './src/components/chat/Header.tsx',
        './src/components/layout/AppLayout.tsx',
        './src/lib/chatUtils.ts',
        './src/hooks/useChat.ts',
        './src/components/landing/OmLogo.tsx',
        './src/components/ui/Toast.tsx',
        './src/api/index.ts',
        './src/utils/secureTokens.ts',
      ],
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
      'zustand/react/shallow',
      'clsx',
      'tailwind-merge',
      // Pre-bundle markdown stack so it's ready when Chat loads
      'react-markdown',
      'remark-gfm',
      'remark-math',
      'rehype-katex',
      'katex',
    ],
    // Exclude heavy deps not used on initial load
    exclude: ['react-syntax-highlighter', 'ogl', 'lucide-react'],
    esbuildOptions: {
      target: 'es2020',
    },
  },
  esbuild: {
    // Drop console/debugger only in production builds
    drop: process.env.NODE_ENV === 'production' ? ['console', 'debugger'] : [],
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
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/__tests__/setup.ts'],
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
  },
})
