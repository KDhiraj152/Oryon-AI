import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store';
import { auth } from '../api';
import {
  ArrowRight,
  Loader2,
  Eye,
  EyeOff,
  AlertCircle
} from 'lucide-react';
import { OmLogo } from '../components/landing/OmLogo';
import { useUIStore } from '../store/uiStore';

// Password strength calculator
const calculatePasswordStrength = (password: string): { score: number; label: string; color: string } => {
  let score = 0;

  if (password.length >= 6) score += 1;
  if (password.length >= 10) score += 1;
  if (/[a-z]/.test(password) && /[A-Z]/.test(password)) score += 1;
  if (/\d/.test(password)) score += 1;
  if (/[^a-zA-Z0-9]/.test(password)) score += 1;

  if (score <= 1) return { score, label: 'Weak', color: 'bg-red-500' };
  if (score <= 2) return { score, label: 'Fair', color: 'bg-orange-500' };
  if (score <= 3) return { score, label: 'Good', color: 'bg-yellow-500' };
  if (score <= 4) return { score, label: 'Strong', color: 'bg-emerald-500' };
  return { score, label: 'Very Strong', color: 'bg-emerald-500' };
};

/** Auth header with logo and title */
function AuthHeader({ isLogin, isDark }: Readonly<{ isLogin: boolean; isDark: boolean }>) {
  return (
    <div className="text-center mb-12">
      <div className="inline-flex items-center justify-center mb-8 transition-transform hover:scale-105 duration-500">
        <OmLogo variant="minimal" size={42} animated={false} color={isDark ? 'dark' : 'light'} />
      </div>
      <h1
        key={isLogin ? 'login' : 'signup'}
        className={`text-3xl font-semibold tracking-tight mb-3 animate-in fade-in slide-in-from-bottom-2 duration-300 ${isDark ? 'text-white' : 'text-[#111]'}`}
      >
        {isLogin ? 'Welcome back' : 'Create account'}
      </h1>
      <p
        key={`desc-${isLogin ? 'login' : 'signup'}`}
        className={`text-[15px] animate-in fade-in slide-in-from-bottom-2 duration-300 ${isDark ? 'text-white/50' : 'text-black/50'}`}
      >
        {isLogin ? 'Enter your details to sign in.' : 'Start your learning journey today.'}
      </p>
    </div>
  );
}

/** Password field with strength indicator */
function PasswordField({ isDark, password, setPassword, showPassword, setShowPassword, isLogin, touched, onBlur, getInputClasses, getLabelClasses, getStrengthBarClass, passwordStrength, passwordValid }: Readonly<{
  isDark: boolean;
  password: string;
  setPassword: (v: string) => void;
  showPassword: boolean;
  setShowPassword: (v: boolean) => void;
  isLogin: boolean;
  touched: boolean;
  onBlur: () => void;
  getInputClasses: (err: boolean) => string;
  getLabelClasses: (err: boolean) => string;
  getStrengthBarClass: (level: number) => string;
  passwordStrength: { score: number };
  passwordValid: boolean;
}>) {
  const autoCompleteValue = isLogin ? 'current-password' : 'new-password';

  return (
    <div className="space-y-1.5">
      <div className="relative group">
        <input
          id="password"
          name="password"
          type={showPassword ? 'text' : 'password'}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onBlur={onBlur}
          placeholder="Password"
          ref={el => { if (el) el.setAttribute('autocomplete', autoCompleteValue); }}
          className={`pr-12 ${getInputClasses(touched && !passwordValid)}`}
        />
        <label htmlFor="password" className={getLabelClasses(touched && !passwordValid)}>
          Password
        </label>
        <button
          type="button"
          onClick={() => setShowPassword(!showPassword)}
          className={`absolute right-3 top-1/2 -translate-y-1/2 p-2 rounded-lg transition-colors focus:outline-none
            ${isDark ? 'text-white/30 hover:text-white/70' : 'text-black/30 hover:text-black/70'}`}
        >
          {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
        </button>
      </div>
      <div className={`grid transition-all duration-300 ease-[cubic-bezier(0.4,0,0.2,1)] ${!isLogin && password ? 'grid-rows-[1fr] opacity-100 pt-2' : 'grid-rows-[0fr] opacity-0 pt-0'}`}>
        <div className="overflow-hidden">
          <div className="flex gap-1">
            {[1, 2, 3, 4, 5].map((level) => (
              <div
                key={level}
                className={`h-1 flex-1 rounded-full transition-all duration-300 ${getStrengthBarClass(level)} ${passwordStrength.score >= level ? 'opacity-100' : 'opacity-30'}`}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/** Auth footer links */
function AuthFooter({ isDark, isLogin, setIsLogin, setError, navigate }: Readonly<{
  isDark: boolean;
  isLogin: boolean;
  setIsLogin: (v: boolean) => void;
  setError: (v: string) => void;
  navigate: (path: string) => void;
}>) {
  return (
    <>
      <p className={`text-center text-[14px] mt-8 ${isDark ? 'text-white/40' : 'text-black/40'}`}>
        {isLogin ? "Don't have an account?" : 'Already have an account?'}{' '}
        <button
          onClick={() => { setIsLogin(!isLogin); setError(''); }}
          className={`font-medium transition-colors ml-1
            ${isDark ? 'text-white hover:text-white/80' : 'text-black hover:text-black/80'}`}
        >
          <span key={isLogin ? 'signup-link' : 'login-link'} className="animate-in fade-in duration-300 inline-block">
            {isLogin ? 'Sign up' : 'Log in'}
          </span>
        </button>
      </p>
      <div className="text-center mt-12">
        <button
          onClick={() => navigate('/')}
          className={`text-xs font-medium transition-colors inline-flex items-center gap-1.5 opacity-50 hover:opacity-100
            ${isDark ? 'text-white' : 'text-black'}`}
        >
          ← Back to home
        </button>
      </div>
      <p className={`text-center text-[11px] mt-8 leading-relaxed
        ${isDark ? 'text-white/30' : 'text-black/30'}`}
      >
        By continuing, you agree to our{' '}
        <button className="underline hover:no-underline">Terms of Service</button>
        {' '}and{' '}
        <button className="underline hover:no-underline">Privacy Policy</button>
      </p>
    </>
  );
}

// ============================================
// MINIMAL AUTH PAGE - ChatGPT/Gemini inspired
// ============================================
export default function Auth() {
  const navigate = useNavigate();
  const { login } = useAuthStore();
  const resolvedTheme = useUIStore((s) => s.resolvedTheme);
  const isDark = resolvedTheme === 'dark';

  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [touched, setTouched] = useState({ email: false, password: false, name: false });

  // Real-time validation
  const emailValid = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  const passwordStrength = calculatePasswordStrength(password);
  const passwordValid = password.length >= 6;
  const nameValid = name.trim().length >= 2;

  // Clear error when switching modes
  useEffect(() => {
    setError('');
    setTouched({ email: false, password: false, name: false });
  }, [isLogin]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    // Mark all fields as touched
    setTouched({ email: true, password: true, name: true });

    // Validate
    if (!emailValid) {
      setError('Please enter a valid email address');
      return;
    }
    if (!passwordValid) {
      setError('Password must be at least 6 characters');
      return;
    }
    if (!isLogin && !nameValid) {
      setError('Please enter your name');
      return;
    }

    setLoading(true);

    try {
      if (isLogin) {
        const response = await auth.login(email, password);
        login(response.user, response.access_token, response.refresh_token);
      } else {
        await auth.register(email, password, name);
        const response = await auth.login(email, password);
        login(response.user, response.access_token, response.refresh_token);
      }
      navigate('/chat');
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Authentication failed';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleBlur = (field: 'email' | 'password' | 'name') => {
    setTouched(prev => ({ ...prev, [field]: true }));
  };

  // Helper for input classes
  const getInputClasses = (hasError: boolean) => {
    const baseClasses = "peer w-full px-4 pt-6 pb-2 rounded-xl border text-[15px] outline-none focus:outline-none focus:ring-0 focus:ring-offset-0 placeholder-transparent";
    const darkClasses = `bg-white/[0.05] border-white/[0.08] text-white focus:border-white/20 focus:bg-white/[0.08] ${hasError ? 'border-red-500/50' : ''}`;
    const lightClasses = `bg-white border-black/[0.08] text-black focus:border-black/20 shadow-sm ${hasError ? 'border-red-500/50' : ''}`;

    return `${baseClasses} ${isDark ? darkClasses : lightClasses}`;
  };

  const getLabelClasses = (hasError: boolean) => {
    const baseClasses = "absolute left-4 top-4 origin-[0] -translate-y-3 scale-75 transform text-[11px] font-semibold uppercase tracking-wider duration-200 pointer-events-none peer-placeholder-shown:translate-y-0 peer-placeholder-shown:scale-100 peer-placeholder-shown:font-medium peer-placeholder-shown:normal-case peer-placeholder-shown:tracking-normal peer-focus:-translate-y-3 peer-focus:scale-75 peer-focus:font-semibold peer-focus:uppercase peer-focus:tracking-wider";
    let colorClasses: string;
    if (isDark) {
      colorClasses = 'text-white/50 peer-focus:text-white/80';
      if (hasError) colorClasses += ' text-red-400';
    } else {
      colorClasses = 'text-black/50 peer-focus:text-black/80';
      if (hasError) colorClasses += ' text-red-500';
    }
    return `${baseClasses} ${colorClasses}`;
  };

  const getStrengthBarClass = (level: number) => {
    const isActive = passwordStrength.score >= level;
    if (isActive) {
      return isDark ? 'bg-white' : 'bg-black';
    }
    return isDark ? 'bg-white/10' : 'bg-black/5';
  };

  return (
    <div className={`min-h-screen flex items-center justify-center px-6 font-sans
      ${isDark
        ? 'bg-[#0a0a0a] text-white'
        : 'bg-[#FAFAFA] text-[#111]'}`}
    >
      {/* Skip link for accessibility */}
      <a href="#auth-form" className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-white text-black px-4 py-2 rounded-md z-50">Skip to form</a>

      <div className="w-full max-w-[380px] animate-enter">

        <AuthHeader isLogin={isLogin} isDark={isDark} />

        {/* Form */}
        <form id="auth-form" onSubmit={handleSubmit} className="space-y-5" noValidate>

          {/* Name (signup only) */}
          <div className={`grid transition-all duration-300 ease-[cubic-bezier(0.4,0,0.2,1)] ${isLogin ? 'grid-rows-[0fr] opacity-0' : 'grid-rows-[1fr] opacity-100'}`}>
            <div className="overflow-hidden">
              <div className="pb-5">
                <div className="relative group">
                  <input
                    id="name"
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    onBlur={() => handleBlur('name')}
                    placeholder="Full Name"
                    autoComplete="name"
                    disabled={isLogin}
                    className={getInputClasses(touched.name && !nameValid)}
                  />
                  <label
                    htmlFor="name"
                    className={getLabelClasses(touched.name && !nameValid)}
                  >
                    Full Name
                  </label>
                </div>
              </div>
            </div>
          </div>

          {/* Email */}
          <div className="space-y-1.5">
            <div className="relative group">
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                onBlur={() => handleBlur('email')}
                placeholder="Email Address"
                autoComplete="email"
                className={getInputClasses(touched.email && !emailValid)}
              />
              <label
                htmlFor="email"
                className={getLabelClasses(touched.email && !emailValid)}
              >
                Email Address
              </label>
            </div>
          </div>

          <PasswordField
            isDark={isDark}
            password={password}
            setPassword={setPassword}
            showPassword={showPassword}
            setShowPassword={setShowPassword}
            isLogin={isLogin}
            touched={touched.password}
            onBlur={() => handleBlur('password')}
            getInputClasses={getInputClasses}
            getLabelClasses={getLabelClasses}
            getStrengthBarClass={getStrengthBarClass}
            passwordStrength={passwordStrength}
            passwordValid={passwordValid}
          />

          {/* Error */}
          {error && (
            <div
              className={`flex items-center gap-3 px-4 py-3 rounded-xl text-sm animate-shake
                ${isDark ? 'bg-red-500/10 text-red-400 border border-red-500/20' : 'bg-red-50 text-red-600 border border-red-100'}`}
              role="alert"
            >
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Submit */}
          <button
            type="submit"
            disabled={loading}
            className={`w-full py-3.5 rounded-full text-[15px] font-semibold tracking-tight
              flex items-center justify-center gap-2 mt-2
              transition-all duration-300 hover:scale-[1.02] active:scale-[0.98]
              ${isDark
                ? 'bg-white text-black hover:bg-gray-100 shadow-[0_0_20px_-5px_rgba(255,255,255,0.3)]'
                : 'bg-[#111] text-white hover:bg-black/90 shadow-[0_10px_20px_-5px_rgba(0,0,0,0.2)]'
              }
              disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100`}
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <span key={isLogin ? 'login-btn' : 'signup-btn'} className="flex items-center gap-2 animate-in fade-in slide-in-from-bottom-1 duration-300">
                {isLogin ? 'Sign In' : 'Create Account'}
                <ArrowRight className="w-4 h-4" />
              </span>
            )}
          </button>
        </form>

        <AuthFooter isDark={isDark} isLogin={isLogin} setIsLogin={setIsLogin} setError={setError} navigate={navigate} />
      </div>
    </div>
  );
}
