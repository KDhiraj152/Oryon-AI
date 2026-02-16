/**
 * Motion System — GPU-accelerated animation primitives with reduced motion support.
 *
 * Design principles:
 * - All animations use transform/opacity only (GPU-composited, no layout thrashing)
 * - Reduced motion preference is respected globally
 * - Stagger utilities for message list animations
 * - Spring-based easing curves for natural feel
 * - 60fps minimum target
 */

// ─── Easing Curves ──────────────────────────────────────────────────────────

/** Custom easing curves (matching tailwind.config.js) */
export const EASING = {
  /** Smooth deceleration — primary UI motion */
  smooth: 'cubic-bezier(0.16, 1, 0.3, 1)',
  /** Soft bounce — micro-interactions (buttons, toggles) */
  bounceSoft: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
  /** Expo out — fast open/close transitions */
  expoOut: 'cubic-bezier(0.19, 1, 0.22, 1)',
  /** Standard ease — fallback */
  standard: 'cubic-bezier(0.4, 0, 0.2, 1)',
  /** Spring — natural object motion */
  spring: 'cubic-bezier(0.175, 0.885, 0.32, 1.275)',
} as const;

// ─── Duration Scale ─────────────────────────────────────────────────────────

/** Duration scale in ms */
export const DURATION = {
  instant: 100,
  fast: 150,
  normal: 200,
  slow: 300,
  slower: 400,
  entrance: 500,
} as const;

// ─── Animation Presets ──────────────────────────────────────────────────────

/**
 * GPU-accelerated animation presets.
 * All use transform + opacity only — no layout properties.
 */
export const PRESET = {
  /** Message appearing in chat (slide up + fade in) */
  messageIn: {
    keyframes: [
      { opacity: 0, transform: 'translateY(12px) scale(0.98)' },
      { opacity: 1, transform: 'translateY(0) scale(1)' },
    ],
    options: { duration: DURATION.slow, easing: EASING.smooth, fill: 'forwards' as const },
  },
  /** Element fading in */
  fadeIn: {
    keyframes: [
      { opacity: 0 },
      { opacity: 1 },
    ],
    options: { duration: DURATION.normal, easing: EASING.standard, fill: 'forwards' as const },
  },
  /** Scale in from center (popovers, tooltips) */
  scaleIn: {
    keyframes: [
      { opacity: 0, transform: 'scale(0.95)' },
      { opacity: 1, transform: 'scale(1)' },
    ],
    options: { duration: DURATION.fast, easing: EASING.bounceSoft, fill: 'forwards' as const },
  },
  /** Slide down from top (dropdowns) */
  slideDown: {
    keyframes: [
      { opacity: 0, transform: 'translateY(-8px)' },
      { opacity: 1, transform: 'translateY(0)' },
    ],
    options: { duration: DURATION.normal, easing: EASING.smooth, fill: 'forwards' as const },
  },
  /** Slide up and out (dismissals) */
  slideOut: {
    keyframes: [
      { opacity: 1, transform: 'translateY(0)' },
      { opacity: 0, transform: 'translateY(-8px)' },
    ],
    options: { duration: DURATION.fast, easing: EASING.standard, fill: 'forwards' as const },
  },
  /** Gentle pulse for loading/thinking states */
  pulse: {
    keyframes: [
      { opacity: 0.4 },
      { opacity: 1 },
      { opacity: 0.4 },
    ],
    options: { duration: 1500, easing: EASING.standard, iterations: Infinity },
  },
  /** Skeleton shimmer */
  shimmer: {
    keyframes: [
      { opacity: 0.3 },
      { opacity: 0.6 },
      { opacity: 0.3 },
    ],
    options: { duration: 2000, easing: EASING.standard, iterations: Infinity },
  },
} as const;

// ─── Stagger Utility ────────────────────────────────────────────────────────

/**
 * Calculate stagger delay for a list item.
 * Caps at maxDelay to prevent long waits for large lists.
 */
export function staggerDelay(index: number, baseMs = 40, maxMs = 300): number {
  return Math.min(index * baseMs, maxMs);
}

// ─── Reduced Motion ─────────────────────────────────────────────────────────

/** Check if user prefers reduced motion (SSR-safe) */
export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined') return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/**
 * Apply a Web Animations API animation respecting reduced motion.
 * Returns the Animation object (or null if reduced motion is active).
 */
export function animate(
  element: Element | null,
  preset: typeof PRESET[keyof typeof PRESET],
  overrides?: Partial<KeyframeAnimationOptions>,
): Animation | null {
  if (!element || prefersReducedMotion()) {
    // Immediately apply end state
    if (element && preset.keyframes.length > 0) {
      const endState = preset.keyframes[preset.keyframes.length - 1];
      if ('opacity' in endState) {
        (element as HTMLElement).style.opacity = String(endState.opacity);
      }
    }
    return null;
  }

  return element.animate([...preset.keyframes] as Keyframe[], {
    ...preset.options,
    ...overrides,
  });
}

// ─── CSS Class Helpers ──────────────────────────────────────────────────────

/**
 * Generate class string that disables animations when reduced motion is preferred.
 * Usage: `className={motionSafe('animate-fadeIn', 'opacity-100')}`
 */
export function motionSafe(animationClass: string, fallbackClass = ''): string {
  return `motion-safe:${animationClass} motion-reduce:${fallbackClass || 'transition-none'}`;
}

// ─── Scroll Utilities ───────────────────────────────────────────────────────

/**
 * Smooth scroll to element with reduced motion fallback.
 */
export function scrollToElement(
  element: Element | null,
  behavior: ScrollBehavior = 'smooth',
): void {
  if (!element) return;
  const resolvedBehavior = prefersReducedMotion() ? 'instant' : behavior;
  element.scrollIntoView({ behavior: resolvedBehavior, block: 'end' });
}
