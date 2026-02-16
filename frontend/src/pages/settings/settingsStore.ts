export interface UserSettings {
  voiceType: string;
  speechSpeed: number;
  autoReadResponses: boolean;
}

export const DEFAULT_SETTINGS: UserSettings = {
  voiceType: 'female',
  speechSpeed: 1,
  autoReadResponses: false,
};

export function loadSettings(): UserSettings {
  try {
    const saved = localStorage.getItem('user-settings');
    return saved ? { ...DEFAULT_SETTINGS, ...JSON.parse(saved) } : DEFAULT_SETTINGS;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(settings: UserSettings) {
  localStorage.setItem('user-settings', JSON.stringify(settings));
}
