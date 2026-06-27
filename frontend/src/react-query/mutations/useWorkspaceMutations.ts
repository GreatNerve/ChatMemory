import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiDelete, apiPost } from "@/lib/api/client";
import {
  AskResponse,
  ChatMessage,
  PersonaChatRequest,
  PersonaChatResponse,
  WorkspaceSummary,
} from "@/lib/api/types";

export function useCreateWorkspaceMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (input: { name: string; file: File }) => {
      const form = new FormData();
      form.append("name", input.name);
      form.append("file", input.file);
      return apiPost<{ workspace: WorkspaceSummary; jobId: string }>("/workspaces", form);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["workspaces"] }),
  });
}

export function useDeleteWorkspaceMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiDelete(`/workspaces/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["workspaces"] }),
  });
}

export function useAskMutation(workspaceId: string) {
  return useMutation({
    mutationFn: (body: {
      question: string;
      speaker?: string;
      dateFrom?: string;
      dateTo?: string;
    }) => apiPost<AskResponse>(`/workspaces/${workspaceId}/ask`, body),
  });
}

export function useTrainPersonaMutation(workspaceId: string, personId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { consent: boolean; forceThin?: boolean; forceRetrain?: boolean }) =>
      apiPost<{ jobId: string; personaStatus: string }>(
        `/workspaces/${workspaceId}/people/${personId}/train`,
        body,
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["person", workspaceId, personId] });
      qc.invalidateQueries({ queryKey: ["people", workspaceId] });
    },
  });
}

export function useCancelTrainMutation(workspaceId: string, personId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiPost<{ personaStatus: string; message: string }>(
        `/workspaces/${workspaceId}/people/${personId}/train/cancel`,
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["person", workspaceId, personId] });
      qc.invalidateQueries({ queryKey: ["people", workspaceId] });
    },
  });
}

export function usePersonaChatMutation(workspaceId: string, personId: string) {
  return useMutation({
    mutationFn: (body: PersonaChatRequest) =>
      apiPost<PersonaChatResponse>(
        `/workspaces/${workspaceId}/people/${personId}/chat`,
        body,
      ),
  });
}
