import { describe, it, expect, beforeEach } from 'vitest';
import { useChatStore } from '../store/chatStore';

describe('chatStore', () => {
  beforeEach(() => {
    useChatStore.setState({
      conversations: [],
      activeConversationId: null,
      messages: [],
      streamingMessage: '',
    });
  });

  it('should create a new conversation', () => {
    const store = useChatStore.getState();
    store.createConversation();
    const state = useChatStore.getState();
    expect(state.conversations).toHaveLength(1);
    expect(state.activeConversationId).toBe(state.conversations[0].id);
  });

  it('should set active conversation', () => {
    const store = useChatStore.getState();
    store.createConversation();
    const id = useChatStore.getState().conversations[0].id;
    store.createConversation();
    store.setActiveConversationId(id);
    expect(useChatStore.getState().activeConversationId).toBe(id);
  });

  it('should start with no active conversation', () => {
    expect(useChatStore.getState().activeConversationId).toBeNull();
    expect(useChatStore.getState().conversations).toHaveLength(0);
  });

  it('should delete a conversation', () => {
    const store = useChatStore.getState();
    store.createConversation();
    const id = useChatStore.getState().conversations[0].id;
    store.deleteConversation(id);
    expect(useChatStore.getState().conversations).toHaveLength(0);
  });
});
