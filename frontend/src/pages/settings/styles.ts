// Style constants to avoid conditional branches
const DARK_FOCUS = 'focus-visible:ring-white focus-visible:ring-offset-[#0a0a0a]';
const LIGHT_FOCUS = 'focus-visible:ring-gray-400 focus-visible:ring-offset-gray-50';

const FOCUS_STYLES = { dark: DARK_FOCUS, light: LIGHT_FOCUS } as const;

export function getFocusStyle(isDark: boolean): string {
  return FOCUS_STYLES[isDark ? 'dark' : 'light'];
}

/** Get toggle button style based on selection and theme */
export function getToggleStyle(isDark: boolean, isSelected: boolean): string {
  if (isSelected)
    return isDark ? 'bg-white text-black' : 'bg-gray-900 text-white';
  return isDark
    ? 'bg-white/5 text-white/60 hover:bg-white/10'
    : 'bg-gray-50 text-gray-600 hover:bg-gray-100';
}
