import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import { useAuthStore } from '../store/authStore';
import type { LoginRequest, RegisterRequest } from '../types/api';

export function useAuth() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { setTokens, setUser, logout: clearAuth, isAuthenticated } = useAuthStore();

  const loginMutation = useMutation({
    mutationFn: (data: LoginRequest) => api.login(data),
    onSuccess: async (response) => {
      setTokens(response.access_token, response.refresh_token);
      
      try {
        const user = await api.getCurrentUser();
        setUser(user);
        queryClient.setQueryData(['user'], user);
      } catch (error) {
        console.error('Failed to fetch user profile:', error);
      }
      
      navigate('/dashboard');
    },
    onError: (error: any) => {
      console.error('Login failed:', error.response?.data?.message || error.message);
    }
  });

  const registerMutation = useMutation({
    mutationFn: (data: RegisterRequest) => api.register(data),
    onSuccess: async (response) => {
      setTokens(response.access_token, response.refresh_token);
      
      try {
        const user = await api.getCurrentUser();
        setUser(user);
        queryClient.setQueryData(['user'], user);
      } catch (error) {
        console.error('Failed to fetch user profile:', error);
      }
      
      navigate('/dashboard');
    },
    onError: (error: any) => {
      console.error('Registration failed:', error.response?.data?.message || error.message);
    }
  });

  const userQuery = useQuery({
    queryKey: ['user'],
    queryFn: () => api.getCurrentUser(),
    enabled: isAuthenticated,
    staleTime: 5 * 60 * 1000,
    retry: 1
  });

  const logout = () => {
    clearAuth();
    queryClient.clear();
    navigate('/login');
  };

  return {
    user: userQuery.data || null,
    isAuthenticated,
    isLoading: loginMutation.isPending || registerMutation.isPending || userQuery.isLoading,
    isError: loginMutation.isError || registerMutation.isError || userQuery.isError,
    
    login: loginMutation.mutate,
    register: registerMutation.mutate,
    logout,
    
    loginError: loginMutation.error,
    registerError: registerMutation.error
  };
}
