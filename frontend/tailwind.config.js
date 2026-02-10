/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: ['class', "class"],
  theme: {
  	extend: {
  		fontFamily: {
  			sans: [
  				'"Plus Jakarta Sans"',
  				'Inter',
  				'system-ui',
  				'-apple-system',
  				'sans-serif'
  			],
        mono: [
          '"JetBrains Mono"',
          'ui-monospace',
          'SFMono-Regular',
          'monospace'
        ]
  		},
  		// ===== DESIGN TOKENS =====
  		spacing: {
  			'touch': '44px', // Minimum touch target (WCAG)
  			'touch-lg': '48px',
  			'18': '4.5rem',
  			'88': '22rem',
  		},
  		fontSize: {
  			'caption': ['11px', { lineHeight: '1.45', letterSpacing: '0.01em' }],
  			'body-sm': ['13px', { lineHeight: '1.5', letterSpacing: '-0.003em' }],
  			'body': ['15px', { lineHeight: '1.6', letterSpacing: '-0.006em' }],
  			'body-lg': ['17px', { lineHeight: '1.5', letterSpacing: '-0.01em' }],
  			'title-sm': ['18px', { lineHeight: '1.35', letterSpacing: '-0.014em' }],
  			'title': ['22px', { lineHeight: '1.3', letterSpacing: '-0.018em' }],
  			'heading': ['28px', { lineHeight: '1.2', letterSpacing: '-0.021em' }],
  			'display-sm': ['36px', { lineHeight: '1.15', letterSpacing: '-0.024em' }],
  			'display': ['48px', { lineHeight: '1.1', letterSpacing: '-0.028em' }],
  			'display-lg': ['64px', { lineHeight: '1.05', letterSpacing: '-0.032em' }],
  		},
  		colors: {
  			noir: '#0a0a0a',
  			graphite: '#1a1a1a',
  			'warm-white': '#FBFBFB',
  			gold: '#F4D47A',
  			// Semantic opacity colors for dark mode
  			'dark-subtle': 'rgba(255, 255, 255, 0.04)',
  			'dark-muted': 'rgba(255, 255, 255, 0.08)',
  			'dark-border': 'rgba(255, 255, 255, 0.1)',
  			'dark-hover': 'rgba(255, 255, 255, 0.12)',
  			'dark-text-tertiary': 'rgba(255, 255, 255, 0.4)',
  			'dark-text-secondary': 'rgba(255, 255, 255, 0.6)',
  			'dark-text-primary': 'rgba(255, 255, 255, 0.9)',
  			// Semantic opacity colors for light mode
  			'light-subtle': 'rgba(0, 0, 0, 0.02)',
  			'light-muted': 'rgba(0, 0, 0, 0.05)',
  			'light-border': 'rgba(0, 0, 0, 0.08)',
  			'light-hover': 'rgba(0, 0, 0, 0.06)',
  			'light-text-tertiary': 'rgba(0, 0, 0, 0.4)',
  			'light-text-secondary': 'rgba(0, 0, 0, 0.6)',
  			'light-text-primary': 'rgba(0, 0, 0, 0.9)',
  			background: 'hsl(var(--background))',
  			foreground: 'hsl(var(--foreground))',
  			card: {
  				DEFAULT: 'hsl(var(--card))',
  				foreground: 'hsl(var(--card-foreground))'
  			},
  			popover: {
  				DEFAULT: 'hsl(var(--popover))',
  				foreground: 'hsl(var(--popover-foreground))'
  			},
  			primary: {
  				DEFAULT: 'hsl(var(--primary))',
  				foreground: 'hsl(var(--primary-foreground))'
  			},
  			secondary: {
  				DEFAULT: 'hsl(var(--secondary))',
  				foreground: 'hsl(var(--secondary-foreground))'
  			},
  			muted: {
  				DEFAULT: 'hsl(var(--muted))',
  				foreground: 'hsl(var(--muted-foreground))'
  			},
  			accent: {
  				DEFAULT: 'hsl(var(--accent))',
  				foreground: 'hsl(var(--accent-foreground))'
  			},
  			destructive: {
  				DEFAULT: 'hsl(var(--destructive))',
  				foreground: 'hsl(var(--destructive-foreground))'
  			},
  			border: 'hsl(var(--border))',
  			input: 'hsl(var(--input))',
  			ring: 'hsl(var(--ring))',
  			chart: {
  				'1': 'hsl(var(--chart-1))',
  				'2': 'hsl(var(--chart-2))',
  				'3': 'hsl(var(--chart-3))',
  				'4': 'hsl(var(--chart-4))',
  				'5': 'hsl(var(--chart-5))'
  			}
  		},
  		borderRadius: {
  			'btn': '10px',
  			'btn-lg': '12px',
  			'card': '16px',
  			'card-lg': '20px',
  			'modal': '24px',
  			lg: 'var(--radius)',
  			md: 'calc(var(--radius) - 2px)',
  			sm: 'calc(var(--radius) - 4px)'
  		},
  		boxShadow: {
  			'glow-sm': '0 0 20px rgba(99, 102, 241, 0.15)',
  			'glow': '0 0 40px rgba(99, 102, 241, 0.2)',
  			'glow-lg': '0 0 60px rgba(99, 102, 241, 0.25)',
  			'float': '0 20px 40px -12px rgba(0, 0, 0, 0.25)',
  			'float-lg': '0 32px 64px -16px rgba(0, 0, 0, 0.35)',
  			'inner-glow': 'inset 0 1px 0 0 rgba(255, 255, 255, 0.05)',
  		},
  		transitionTimingFunction: {
  			'smooth': 'cubic-bezier(0.16, 1, 0.3, 1)',
  			'bounce-soft': 'cubic-bezier(0.34, 1.56, 0.64, 1)',
  			'out-expo': 'cubic-bezier(0.19, 1, 0.22, 1)',
  		},
  		transitionDuration: {
  			'fast': '150ms',
  			'normal': '200ms',
  			'slow': '300ms',
  			'slower': '400ms',
  		},
  		animation: {
  			'breathe': 'breathe 4s ease-in-out infinite',
  			'float': 'float 6s ease-in-out infinite',
  			'glow': 'glow-pulse 3s ease-in-out infinite',
  			'drift': 'drift 20s ease-in-out infinite',
  			'pulse-soft': 'pulse-soft 3s ease-in-out infinite',
  			'fadeInUp': 'fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards',
  			'fadeIn': 'fadeIn 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards',
  			'slideUp': 'slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards',
  			'slideDown': 'slideDown 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards',
  			'scaleIn': 'scaleIn 0.2s cubic-bezier(0.34, 1.56, 0.64, 1) forwards',
  			'shake': 'shake 0.5s cubic-bezier(0.36, 0.07, 0.19, 0.97) both',
  			'spin-slow': 'spin 8s linear infinite',
  		},
  		keyframes: {
  			slideUp: {
  				'0%': { opacity: '0', transform: 'translateY(10px)' },
  				'100%': { opacity: '1', transform: 'translateY(0)' },
  			},
  			slideDown: {
  				'0%': { opacity: '0', transform: 'translateY(-10px)' },
  				'100%': { opacity: '1', transform: 'translateY(0)' },
  			},
  			scaleIn: {
  				'0%': { opacity: '0', transform: 'scale(0.95)' },
  				'100%': { opacity: '1', transform: 'scale(1)' },
  			},
  			shake: {
  				'10%, 90%': { transform: 'translateX(-1px)' },
  				'20%, 80%': { transform: 'translateX(2px)' },
  				'30%, 50%, 70%': { transform: 'translateX(-3px)' },
  				'40%, 60%': { transform: 'translateX(3px)' },
  			},
  		},
  		// Z-index scale
  		zIndex: {
  			'dropdown': '100',
  			'sticky': '200',
  			'overlay': '300',
  			'modal': '400',
  			'toast': '500',
  		},
  	}
  },
  plugins: [
    require('@tailwindcss/typography'),
    require("tailwindcss-animate"),
    // Custom plugin for focus-visible ring
    function({ addUtilities }) {
      addUtilities({
        '.focus-ring': {
          '@apply focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-400 focus-visible:ring-offset-2 focus-visible:ring-offset-noir': {},
        },
        '.focus-ring-light': {
          '@apply focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-400 focus-visible:ring-offset-2 focus-visible:ring-offset-white': {},
        },
      })
    },
  ],
}
