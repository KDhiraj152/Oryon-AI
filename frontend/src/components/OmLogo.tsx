// ============================================
// SACRED OM SYMBOL (ॐ)
// Using actual Devanagari Om character for authenticity
// ============================================

interface OmLogoProps {
  size?: number;
  className?: string;
  animated?: boolean;
  color?: 'light' | 'dark' | 'auto';
}

export const OmLogo = ({
  size = 80,
  className = '',
  animated = true,
  color = 'auto'
}: OmLogoProps) => {
  // Determine fill color based on mode
  const getFillColor = () => {
    if (color === 'light') return '#0a0a0a';
    if (color === 'dark') return '#ffffff';
    return 'currentColor';
  };
  const fillColor = getFillColor();

  return (
    <svg
      viewBox="0 0 100 100"
      width={size}
      height={size}
      className={`${className} ${animated ? 'animate-glow' : ''}`}
      aria-label="Om symbol"
    >
      <defs>
        {/* Subtle glow filter */}
        <filter id="omGlow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feFlood floodColor="#5161FF" floodOpacity="0.3" />
          <feComposite in2="blur" operator="in" />
          <feMerge>
            <feMergeNode />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Actual Om character (ॐ) - authentic Devanagari */}
      <text
        x="50"
        y="68"
        textAnchor="middle"
        fontSize="72"
        fontFamily="'Noto Sans Devanagari', 'Mangal', 'Sanskrit Text', serif"
        fill={fillColor}
        filter={animated ? "url(#omGlow)" : undefined}
        opacity="0.95"
      >
        ॐ
      </text>
    </svg>
  );
};

// Simple Om for inline use
export const OmIcon = ({ size = 24, className = '' }: { size?: number; className?: string }) => (
  <svg
    viewBox="0 0 24 24"
    width={size}
    height={size}
    className={className}
    fill="currentColor"
    aria-label="Om"
  >
    <text
      x="12"
      y="18"
      textAnchor="middle"
      fontSize="20"
      fontFamily="Arial Unicode MS, Noto Sans Devanagari, sans-serif"
    >
      ॐ
    </text>
  </svg>
);

export default OmLogo;
