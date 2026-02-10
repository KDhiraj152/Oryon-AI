import { useEffect, useState } from 'react';

interface OmLogoProps {
  size?: number;
  className?: string;
  animated?: boolean;
  variant?: 'hero' | 'nav' | 'footer' | 'minimal';
  color?: 'light' | 'dark';
}

// Rotating taglines in different Indian languages
const taglines = [
  { text: 'सबके लिए, सबकी भाषा में', lang: 'Hindi' },
  { text: 'எல்லோருக்கும், அவரவர் மொழியில்', lang: 'Tamil' },
  { text: 'అందరికీ, వారి భాషలో', lang: 'Telugu' },
  { text: 'সবার জন্য, নিজের ভাষায়', lang: 'Bengali' },
  { text: 'ಎಲ್ಲರಿಗೂ, ಅವರವರ ಭಾಷೆಯಲ್ಲಿ', lang: 'Kannada' },
  { text: 'എല്ലാവർക്കും, സ്വന്തം ഭാഷയിൽ', lang: 'Malayalam' },
  { text: 'सगळ्यांसाठी, आपल्या भाषेत', lang: 'Marathi' },
  { text: 'ਸਭ ਲਈ, ਆਪਣੀ ਭਾਸ਼ਾ ਵਿੱਚ', lang: 'Punjabi' },
  { text: 'બધા માટે, પોતાની ભાષામાં', lang: 'Gujarati' },
  { text: 'ସମସ୍ତଙ୍କ ପାଇଁ, ନିଜ ଭାଷାରେ', lang: 'Odia' },
  { text: 'سب کستہ، اپنی زبان ميں', lang: 'Urdu' },
  { text: 'সকলোৰে বাবে, নিজৰ ভাষাত', lang: 'Assamese' },
];

// Custom hook for rotating taglines (exported for use in landing page)
export const useRotatingTagline = () => {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % taglines.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return taglines[index];
};

// Beautiful Om symbol using the actual Unicode character with proper font
const OmSymbol = ({
  size = 48,
  color = '#ffffff',
  className = ''
}: {
  size?: number;
  color?: string;
  className?: string;
}) => (
  <div
    className={`flex items-center justify-center select-none ${className}`}
    style={{
      width: size,
      height: size,
      fontSize: size * 0.85,
      fontFamily: "'Noto Sans Devanagari', 'Arial Unicode MS', serif",
      fontWeight: 400,
      color: color,
      lineHeight: 1,
    }}
  >
    ॐ
  </div>
);

interface VariantProps {
  size: number;
  className: string;
  animated: boolean;
  color: 'light' | 'dark';
  isVisible: boolean;
  textColor: string;
  glowColor: string;
  accentColor: string;
}

function HeroVariant({ size, className, animated, color, isVisible, textColor, glowColor, accentColor }: Readonly<VariantProps>) {
  const borderColor = color === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)';
  const pulseAnim = animated ? 'pulse 4s ease-in-out infinite' : 'none';
  return (
    <div
      className={`relative flex items-center justify-center ${className} ${isVisible ? 'opacity-100' : 'opacity-0'} transition-opacity duration-700`}
      style={{ width: size, height: size }}
    >
      {animated && (
        <div
          className="absolute rounded-full"
          style={{
            width: size * 0.85,
            height: size * 0.85,
            border: `1px solid ${borderColor}`,
            animation: 'spin 30s linear infinite',
          }}
        />
      )}
      <div
        className="absolute rounded-full"
        style={{
          width: size * 0.6,
          height: size * 0.6,
          background: `radial-gradient(circle, ${accentColor} 0%, transparent 70%)`,
          filter: 'blur(20px)',
          animation: pulseAnim,
        }}
      />
      <div
        className={animated ? 'animate-breathe' : ''}
        style={{ filter: `drop-shadow(0 0 20px ${glowColor}30)` }}
      >
        <OmSymbol size={size * 0.45} color={textColor} />
      </div>
    </div>
  );
}

function NavVariant({ size, className, color, isVisible, textColor, glowColor }: Readonly<VariantProps>) {
  const filterVal = color === 'dark' ? `drop-shadow(0 0 6px ${glowColor}20)` : 'none';
  return (
    <div className={`flex items-center gap-2.5 ${className} ${isVisible ? 'opacity-100' : 'opacity-0'} transition-opacity duration-500`}>
      <div style={{ filter: filterVal }}>
        <OmSymbol size={size} color={textColor} />
      </div>
      <span className="text-sm font-semibold tracking-tight" style={{ color: textColor }}>
        shiksha setu
      </span>
    </div>
  );
}

function FooterVariant({ size, className, color, isVisible }: Readonly<VariantProps>) {
  const symbolColor = color === 'dark' ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.4)';
  const dividerBg = color === 'dark' ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.1)';
  const labelColor = color === 'dark' ? 'rgba(255,255,255,0.35)' : 'rgba(0,0,0,0.35)';
  return (
    <div className={`flex items-center gap-3 ${className} ${isVisible ? 'opacity-100' : 'opacity-0'} transition-opacity duration-500`}>
      <OmSymbol size={size} color={symbolColor} />
      <div className="h-4 w-px" style={{ background: dividerBg }} />
      <span className="text-xs tracking-widest uppercase font-medium" style={{ color: labelColor }}>
        Shiksha Setu
      </span>
    </div>
  );
}

export const OmLogo: React.FC<OmLogoProps> = ({
  size = 200,
  className = '',
  animated = true,
  variant = 'hero',
  color = 'dark'
}) => {
  // Only use fade-in animation for hero variant, others show immediately
  const [isVisible, setIsVisible] = useState(variant !== 'hero');

  useEffect(() => {
    if (variant === 'hero') {
      const timer = setTimeout(() => setIsVisible(true), 50);
      return () => clearTimeout(timer);
    }
  }, [variant]);

  const textColor = color === 'dark' ? '#ffffff' : '#1a1a1a';
  const glowColor = '#6366f1';
  const accentColor = color === 'dark' ? 'rgba(99, 102, 241, 0.15)' : 'rgba(99, 102, 241, 0.08)';

  const props: VariantProps = { size, className, animated, color, isVisible, textColor, glowColor, accentColor };

  if (variant === 'hero') return <HeroVariant {...props} />;
  if (variant === 'nav') return <NavVariant {...props} />;
  if (variant === 'footer') return <FooterVariant {...props} />;

  // === MINIMAL VARIANT ===
  return (
    <div className={`flex items-center justify-center ${className} ${isVisible ? 'opacity-100' : 'opacity-0'} transition-opacity duration-500`}>
      <OmSymbol size={size} color={textColor} />
    </div>
  );
};

export default OmLogo;
