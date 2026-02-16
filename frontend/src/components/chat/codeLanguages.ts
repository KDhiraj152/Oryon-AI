/**
 * Language display name mappings for code blocks.
 * Extracted to a separate file to comply with React Fast Refresh rules
 * (a file should only export components, or only export non-components).
 */
export const LANG_DISPLAY: Record<string, string> = {
  js: 'JavaScript', javascript: 'JavaScript',
  ts: 'TypeScript', typescript: 'TypeScript',
  py: 'Python', python: 'Python',
  jsx: 'JSX', tsx: 'TSX',
  html: 'HTML', css: 'CSS', scss: 'SCSS',
  json: 'JSON', yaml: 'YAML', yml: 'YAML',
  sql: 'SQL', bash: 'Bash', sh: 'Shell',
  c: 'C', cpp: 'C++', java: 'Java',
  go: 'Go', rust: 'Rust', ruby: 'Ruby',
  php: 'PHP', swift: 'Swift', kotlin: 'Kotlin',
  md: 'Markdown', markdown: 'Markdown',
};
