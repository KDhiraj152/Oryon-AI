import { createContext, useContext, useEffect, useMemo, useCallback } from 'react';
import { useThemeStore } from '../store';

type Theme = 'light' | 'dark';

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const ThemeProvider = ({ children }: { children: React.ReactNode }) => {
  const { resolvedTheme, setTheme: setStoreTheme } = useThemeStore();

  // Sync DOM with store on mount and updates
  useEffect(() => {
    const root = document.documentElement;
    if (resolvedTheme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [resolvedTheme]);

  const toggleTheme = useCallback(() => {
    setStoreTheme(resolvedTheme === 'dark' ? 'light' : 'dark');
  }, [resolvedTheme, setStoreTheme]);

  const setTheme = useCallback((newTheme: Theme) => {
    setStoreTheme(newTheme);
  }, [setStoreTheme]);

  const contextValue = useMemo(() => ({
    theme: resolvedTheme,
    toggleTheme,
    setTheme
  }), [resolvedTheme, toggleTheme, setTheme]);

  return (
    <ThemeContext.Provider value={contextValue}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};
