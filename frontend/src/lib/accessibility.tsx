import { useState, useEffect } from 'react';

/**
 * SkipLink component - Accessibility feature for keyboard navigation
 * Allows users to skip to main content using keyboard shortcut
 * Typically appears as a link that's visible on focus
 */
export function SkipLink() {
  const [isFocused, setIsFocused] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Skip to main content on Alt+S or using tab
      if ((e.altKey && e.key === 's') || (e.key === 'Tab' && !isFocused)) {
        setIsFocused(true);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsFocused(false);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('keyup', handleKeyUp);
    };
  }, [isFocused]);

  const handleClick = () => {
    const mainContent = document.querySelector('main') || document.querySelector('[role="main"]');
    if (mainContent) {
      (mainContent as HTMLElement).focus();
      (mainContent as HTMLElement).scrollIntoView({ behavior: 'smooth' });
    }
    setIsFocused(false);
  };

  return (
    <a
      href="#main-content"
      onClick={handleClick}
      className={`
        fixed top-0 left-0 z-50 px-4 py-2 
        bg-blue-600 text-white font-semibold 
        rounded-b-md shadow-lg
        transform transition-opacity duration-200
        ${isFocused ? 'opacity-100' : 'opacity-0 pointer-events-none'}
        focus:opacity-100 focus:pointer-events-auto
      `}
      onFocus={() => setIsFocused(true)}
      onBlur={() => setIsFocused(false)}
    >
      Skip to main content (Alt+S)
    </a>
  );
}

/**
 * Add role="main" to your main content area for proper accessibility
 * Example: <main role="main" id="main-content">
 */
