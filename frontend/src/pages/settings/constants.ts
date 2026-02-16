import { User, UserCircle, Sparkles, BookOpen, FlaskConical, Lock } from 'lucide-react';

export const VOICE_TYPES = [
  { value: 'female', label: 'Female', Icon: User },
  { value: 'male', label: 'Male', Icon: UserCircle },
] as const;

export const SPEECH_SPEEDS = [
  { value: 0.75, label: '0.75×', desc: 'Slow' },
  { value: 1, label: '1×', desc: 'Normal' },
  { value: 1.25, label: '1.25×', desc: 'Fast' },
  { value: 1.5, label: '1.5×', desc: 'Faster' },
] as const;

export type PolicyMode = 'OPEN' | 'MODERATED' | 'RESEARCH' | 'RESTRICTED';

export const POLICY_MODES: {
  value: PolicyMode;
  label: string;
  desc: string;
  icon: typeof Sparkles;
  color: string;
}[] = [
  {
    value: 'OPEN',
    label: 'Open',
    desc: 'General AI with essential safety',
    icon: Sparkles,
    color: 'emerald',
  },
  {
    value: 'MODERATED',
    label: 'Education',
    desc: 'content domain aligned',
    icon: BookOpen,
    color: 'blue',
  },
  {
    value: 'RESEARCH',
    label: 'Research',
    desc: 'Maximum freedom for academics',
    icon: FlaskConical,
    color: 'purple',
  },
  {
    value: 'RESTRICTED',
    label: 'Restricted',
    desc: 'Full policy enforcement',
    icon: Lock,
    color: 'amber',
  },
];
