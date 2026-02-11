import { useState, useEffect, memo, useMemo, lazy, Suspense } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Sun, Moon, Mic, Languages, Brain, Zap, Camera, Sparkles, Globe } from 'lucide-react';
import { OmLogo } from '../components/landing/OmLogo';
import { useTheme } from '../context/ThemeContext';
import LogoLoop from '../components/LogoLoop';

// Lazy load heavy WebGL component
const LightRays = lazy(() => import('../components/LightRays'));

interface SectionProps {
  isDark: boolean;
}

const cardClass = (isDark: boolean) =>
  `p-8 rounded-[2rem] border transition-all duration-500 hover:scale-[1.01] group ${isDark ? 'bg-[#111] border-white/[0.05] hover:border-white/[0.1]' : 'bg-white border-black/[0.04] shadow-[0_2px_20px_rgba(0,0,0,0.02)] hover:shadow-[0_10px_30px_rgba(0,0,0,0.06)]'}`;

function LandingHeader({ isDark }: Readonly<SectionProps>) {
  const navigate = useNavigate();
  const { toggleTheme } = useTheme();
  return (
    <header className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500
      ${isDark ? 'bg-[#0a0a0a]/50 border-b border-white/[0.03]' : 'bg-[#FAFAFA]/70 border-b border-black/[0.03]'} backdrop-blur-xl`}>
      <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
        <button
          type="button"
          className="flex items-center gap-2 group cursor-pointer bg-transparent border-none p-0"
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
        >
          <OmLogo variant="minimal" size={32} color={isDark ? 'dark' : 'light'} animated={false} />
          <span className={`font-bold text-lg tracking-tight ${isDark ? 'text-white' : 'text-black'}`}>Shiksha Setu</span>
        </button>
        <div className="flex items-center gap-4">
          <button
            onClick={toggleTheme}
            className={`p-2.5 rounded-full transition-all duration-300 hover:rotate-12
              ${isDark ? 'hover:bg-white/10 text-white/60 hover:text-white' : 'hover:bg-black/5 text-black/40 hover:text-black'}`}
          >
            {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
          <button
            onClick={() => navigate('/auth')}
            className={`px-6 py-2.5 text-sm font-semibold rounded-full transition-all duration-300 hover:scale-105 active:scale-95
              ${isDark
                ? 'bg-white text-black hover:bg-gray-100 shadow-[0_0_20px_-5px_rgba(255,255,255,0.3)]'
                : 'bg-[#111] text-white hover:bg-black/90 shadow-[0_4px_15px_-3px_rgba(0,0,0,0.2)]'
              }`}
          >
            Get Started
          </button>
        </div>
      </div>
    </header>
  );
}

function FeaturesSection({ isDark }: Readonly<SectionProps>) {
  const audioBarData = useMemo(
    () => Array.from({ length: 12 }, (_, i) => ({
      id: `audio-bar-${i}`,
      height: `${Math.random() * 60 + 20}%`,
      delay: `${i * 0.1}s`,
    })),
    []
  );
  return (
    <section id="features" className="py-24 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className={`text-3xl md:text-4xl font-bold tracking-tight mb-4 ${isDark ? 'text-white' : 'text-black'}`}>
            Powered by 8 AI Models
          </h2>
          <p className={`text-lg font-medium max-w-xl mx-auto ${isDark ? 'text-white/50' : 'text-black/50'}`}>
            Enterprise-grade technology, built for everyone.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className={`md:col-span-2 ${cardClass(isDark)}`}>
            <div className="flex items-start justify-between mb-6">
              <div className={`p-3 rounded-xl ${isDark ? 'bg-white/10 text-white' : 'bg-gray-100 text-black'}`}><Camera className="w-6 h-6" /></div>
              <span className={`px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider ${isDark ? 'bg-white/10 text-white/70' : 'bg-black/5 text-black/70'}`}>GOT-OCR2</span>
            </div>
            <h3 className="text-2xl font-bold mb-3">Vision Intelligence</h3>
            <p className={`text-base leading-relaxed max-w-sm ${isDark ? 'text-white/60' : 'text-black/60'}`}>
              Upload photos of handwritten notes, complex diagrams, or textbook pages. Our AI sees and understands everything instantly.
            </p>
          </div>
          <div className={`md:row-span-2 flex flex-col justify-between ${cardClass(isDark)}`}>
            <div>
              <div className={`p-3 rounded-xl w-fit mb-6 ${isDark ? 'bg-white/10 text-white' : 'bg-gray-100 text-black'}`}><Mic className="w-6 h-6" /></div>
              <h3 className="text-2xl font-bold mb-3">Whisper V3 Turbo</h3>
              <p className={`text-base leading-relaxed ${isDark ? 'text-white/60' : 'text-black/60'}`}>
                Record lectures or upload videos. We transcribe and translate speech in real-time with 99% accuracy.
              </p>
            </div>
            <div className={`mt-8 h-24 rounded-xl w-full overflow-hidden relative ${isDark ? 'bg-white/5' : 'bg-gray-50'}`}>
              <div className="absolute inset-0 flex items-center justify-center gap-1">
                {audioBarData.map((bar) => (
                  <div key={bar.id}
                       ref={el => { if (el) { el.style.height = bar.height; el.style.animationDelay = bar.delay; } }}
                       className={`w-1.5 rounded-full animate-pulse ${isDark ? 'bg-white/40' : 'bg-black/20'}`} />
                ))}
              </div>
            </div>
          </div>
          <div className={cardClass(isDark)}>
            <div className="flex items-start justify-between mb-6">
              <div className={`p-3 rounded-xl ${isDark ? 'bg-white/10 text-white' : 'bg-gray-100 text-black'}`}><Languages className="w-6 h-6" /></div>
            </div>
            <h3 className="text-xl font-bold mb-2">IndicTrans2</h3>
            <p className={`text-sm leading-relaxed ${isDark ? 'text-white/60' : 'text-black/60'}`}>
              Native support for 22 Indian languages, preserving cultural context.
            </p>
          </div>
          <div className={cardClass(isDark)}>
            <div className="flex items-start justify-between mb-6">
              <div className={`p-3 rounded-xl ${isDark ? 'bg-white/10 text-white' : 'bg-gray-100 text-black'}`}><Brain className="w-6 h-6" /></div>
            </div>
            <h3 className="text-xl font-bold mb-2">Qwen3-8B</h3>
            <p className={`text-sm leading-relaxed ${isDark ? 'text-white/60' : 'text-black/60'}`}>
              Advanced reasoning that simplifies complex topics to your level.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

function StatsSection({ isDark }: Readonly<SectionProps>) {
  const stats = useMemo(() => [
    { label: 'Languages', value: '22', icon: <Globe className="w-4 h-4" /> },
    { label: 'AI Models', value: '8', icon: <Brain className="w-4 h-4" /> },
    { label: 'Latency', value: '<2s', icon: <Zap className="w-4 h-4" /> },
    { label: 'Cost', value: '₹0', icon: <Sparkles className="w-4 h-4" /> },
  ], []);
  return (
    <section className={`py-24 px-6 border-y ${isDark ? 'border-white/[0.05]' : 'border-black/[0.05]'}`}>
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {stats.map((stat) => (
            <div key={stat.label} className="text-center group">
              <div className={`flex items-center justify-center gap-2 mb-3 opacity-50 group-hover:opacity-100 transition-opacity ${isDark ? 'text-white' : 'text-black'}`}>
                {stat.icon}
                <span className="text-xs font-bold uppercase tracking-wider">{stat.label}</span>
              </div>
              <div className={`text-4xl md:text-5xl font-bold tracking-tighter ${isDark ? 'text-white' : 'text-black'}`}>
                {stat.value}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function LandingFooter({ isDark }: Readonly<SectionProps>) {
  return (
    <footer className={`py-12 px-6 ${isDark ? 'border-t border-white/[0.05]' : 'border-t border-black/[0.05]'}`}>
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-8">
        <div className="flex items-center gap-3 opacity-50 hover:opacity-100 transition-opacity">
          <OmLogo variant="minimal" size={24} color={isDark ? 'dark' : 'light'} animated={false} />
          <span className={`text-sm font-medium ${isDark ? 'text-white' : 'text-black'}`}>© 2025 Shiksha Setu</span>
        </div>
        <div className="flex gap-8">
          {['Privacy', 'Terms', 'Contact'].map((link) => (
            <button
              key={link}
              type="button"
              className={`text-sm font-medium transition-colors bg-transparent border-none p-0 cursor-pointer ${isDark ? 'text-white/40 hover:text-white' : 'text-black/40 hover:text-black'}`}
            >
              {link}
            </button>
          ))}
        </div>
      </div>
    </footer>
  );
}

export const LandingPage = memo(function LandingPage() {
  const navigate = useNavigate();
  const { theme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const isDark = theme === 'dark';

  // Memoize static language logos data
  const languageLogos = useMemo(() => [
    { node: <span className="text-base sm:text-lg font-normal tracking-wide font-sans">हिंदी</span>, title: "Hindi" },
    { node: <span className="text-base sm:text-lg font-normal tracking-wide font-sans">தமிழ்</span>, title: "Tamil" },
    { node: <span className="text-base sm:text-lg font-normal tracking-wide font-sans">తెలుగు</span>, title: "Telugu" },
    { node: <span className="text-base sm:text-lg font-normal tracking-wide font-sans">বাংলা</span>, title: "Bengali" },
    { node: <span className="text-base sm:text-lg font-normal tracking-wide font-sans">मराठी</span>, title: "Marathi" },
    { node: <span className="text-base sm:text-lg font-normal tracking-wide font-sans">ગુજરાતી</span>, title: "Gujarati" },
    { node: <span className="text-base sm:text-lg font-normal tracking-wide font-sans">ಕನ್ನಡ</span>, title: "Kannada" },
    { node: <span className="text-base sm:text-lg font-normal tracking-wide font-sans">മലയാളം</span>, title: "Malayalam" },
    { node: <span className="text-base sm:text-lg font-normal tracking-wide font-sans">ਪੰਜਾਬੀ</span>, title: "Punjabi" },
    { node: <span className="text-base sm:text-lg font-normal tracking-wide font-sans">ଓଡ଼ିଆ</span>, title: "Odia" },
    { node: <span className="text-base sm:text-lg font-normal tracking-wide font-sans">اردو</span>, title: "Urdu" },
    { node: <span className="text-base sm:text-lg font-normal tracking-wide font-sans">অসমীয়া</span>, title: "Assamese" },
    { node: <span className="text-base sm:text-lg font-normal tracking-wide font-sans">संस्कृत</span>, title: "Sanskrit" },
    { node: <span className="text-base sm:text-lg font-normal tracking-wide font-sans">English</span>, title: "English" },
  ], []);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <div className={`min-h-screen font-sans selection:bg-gray-500/30 ${isDark
      ? 'bg-[#0a0a0a] text-white'
      : 'bg-[#FAFAFA] text-[#111]'}`}
    >
      {/* Ambient Background */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className={`absolute top-[-20%] left-1/2 -translate-x-1/2 w-[1000px] h-[1000px] rounded-full blur-[120px] opacity-20
          ${isDark ? 'bg-white/5' : 'bg-gray-400/20'}`} />
        <div className={`absolute bottom-[-20%] right-[-10%] w-[800px] h-[800px] rounded-full blur-[100px] opacity-10
          ${isDark ? 'bg-white/5' : 'bg-gray-400/20'}`} />
      </div>

      {/* LightRays Background - Only in dark mode, lazy loaded */}
      {isDark && (
        <Suspense fallback={null}>
          <div className="fixed inset-0 pointer-events-none z-0 opacity-40">
            <LightRays
              raysOrigin="top-center"
              raysColor="#525252"
              raysSpeed={0.2}
              lightSpread={0.6}
              rayLength={1}
              followMouse={true}
              mouseInfluence={0.05}
              noiseAmount={0.08}
              distortion={0.05}
              fadeDistance={0.5}
              saturation={0.5}
              pulsating={true}
            />
          </div>
        </Suspense>
      )}

      <LandingHeader isDark={isDark} />

      {/* Hero */}
      <main className="relative pt-32 pb-16 px-6 flex flex-col items-center justify-center min-h-[80vh] animate-enter">
        <div className="max-w-4xl mx-auto text-center relative z-10">
          {/* Badge */}
          <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-medium mb-8 backdrop-blur-md border transition-all duration-300 hover:scale-105 cursor-default
            ${isDark ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' : 'bg-emerald-50 border-emerald-200 text-emerald-700 shadow-sm'
          }`}>
            <Sparkles className="w-3 h-3" />
            <span>Open AI • Safe & Unrestricted</span>
          </div>

          {/* Main heading */}
          <h1 className={`text-5xl sm:text-6xl md:text-7xl font-bold tracking-[-0.03em] leading-[1.1] mb-6
            ${isDark ? 'text-white' : 'text-[#111]'}`}>
            AI for <br />
            <span className={`bg-clip-text text-transparent bg-gradient-to-b ${isDark ? 'from-white via-white to-white/50' : 'from-black via-black to-black/40'}`}>
              noble purposes
            </span>
          </h1>

          {/* Subtitle */}
          <p className={`text-lg sm:text-xl font-medium leading-relaxed mb-10 max-w-xl mx-auto tracking-tight
            ${isDark ? 'text-white/50' : 'text-black/60'}`}>
            Safe AI without restrictions. <br className="hidden sm:block" />
            Built for research, creativity, and beyond — in 22 Indian languages.
          </p>

          {/* CTA */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button
              onClick={() => navigate('/chat')}
              className={`group relative inline-flex items-center gap-2 px-6 py-3 rounded-full text-sm font-semibold tracking-tight transition-all duration-300 hover:scale-105 hover:shadow-lg
                ${isDark
                  ? 'bg-white text-black shadow-[0_0_20px_-5px_rgba(255,255,255,0.3)]'
                  : 'bg-[#111] text-white shadow-[0_10px_20px_-5px_rgba(0,0,0,0.2)]'
                }`}
            >
              Get Started
              <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
            </button>

            <button
              onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
              className={`px-6 py-3 rounded-full text-sm font-medium transition-all duration-300 hover:scale-105
                ${isDark
                  ? 'text-white/70 hover:text-white bg-white/[0.05] hover:bg-white/[0.1]'
                  : 'text-black/70 hover:text-black bg-white shadow-sm border border-black/[0.05] hover:bg-gray-50'
                }`}
            >
              How it works
            </button>
          </div>
        </div>

        {/* Language Loop */}
        <div className="absolute bottom-8 left-0 right-0 opacity-60 hover:opacity-100 transition-opacity duration-500">
          <LogoLoop
            logos={languageLogos}
            speed={40}
            direction="left"
            logoHeight={24}
            gap={48}
            hoverSpeed={10}
            fadeOut
            fadeOutColor={isDark ? '#0a0a0a' : '#ffffff'}
            className={isDark ? 'text-white/40' : 'text-black/40'}
          />
        </div>
      </main>

      <FeaturesSection isDark={isDark} />
      <StatsSection isDark={isDark} />

      {/* Final CTA */}
      <section className="py-32 px-6 text-center">
        <div className="max-w-2xl mx-auto">
          <h2 className={`text-4xl md:text-5xl font-bold tracking-tighter mb-6 ${isDark ? 'text-white' : 'text-black'}`}>
            Knowledge is a right.
          </h2>
          <p className={`text-lg md:text-xl font-medium mb-10 ${isDark ? 'text-white/50' : 'text-black/50'}`}>
            No paywalls. No restrictions. Just possibilities.
          </p>
          <button
            onClick={() => navigate('/auth')}
            className={`group inline-flex items-center gap-2 px-8 py-4 rounded-full text-base font-semibold tracking-tight transition-all duration-300 hover:scale-105
              ${isDark
                ? 'bg-white text-black hover:bg-gray-100'
                : 'bg-black text-white hover:bg-gray-900'
              }`}
          >
            Get Started Free
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      </section>

      <LandingFooter isDark={isDark} />
    </div>
  );
});

export default LandingPage;
