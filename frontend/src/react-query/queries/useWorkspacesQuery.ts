import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiGet } from "@/lib/api/client";
import {
  Health,
  PersonDetail,
  PersonSummary,
  Settings,
  SystemStatus,
  WorkspaceAnalytics,
  WorkspaceDetail,
  WorkspaceSummary,
} from "@/lib/api/types";

export function useWorkspacesQuery() {
  return useQuery({
    queryKey: ["workspaces"],
    queryFn: () => apiGet<{ workspaces: WorkspaceSummary[] }>("/workspaces"),
  });
}

export function useWorkspaceQuery(id: string) {
  return useQuery({
    queryKey: ["workspace", id],
    queryFn: () => apiGet<WorkspaceDetail>(`/workspaces/${id}`),
    enabled: Boolean(id),
  });
}

export function usePeopleQuery(workspaceId: string) {
  return useQuery({
    queryKey: ["people", workspaceId],
    queryFn: () => apiGet<{ people: PersonSummary[] }>(`/workspaces/${workspaceId}/people`),
    enabled: Boolean(workspaceId),
    staleTime: 30_000,
  });
}

export function usePersonQuery(workspaceId: string, personId: string) {
  return useQuery({
    queryKey: ["person", workspaceId, personId],
    queryFn: () => apiGet<PersonDetail>(`/workspaces/${workspaceId}/people/${personId}`),
    enabled: Boolean(workspaceId && personId),
    staleTime: 30_000,
  });
}

export function useSettingsQuery() {
  return useQuery({
    queryKey: ["settings"],
    queryFn: () => apiGet<Settings>("/settings"),
  });
}

export function useHealthQuery() {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => apiGet<Health>("/health"),
    refetchInterval: 15_000,
  });
}

/**
 * Poll GET /system/status every 3 s until the embed model is ready, then
 * slow down to once a minute (the model stays warm after first load).
 */
export function useSystemStatusQuery() {
  return useQuery({
    queryKey: ["system-status"],
    queryFn: () => apiGet<SystemStatus>("/system/status"),
    // Fast poll while loading; slow down once ready to reduce noise.
    refetchInterval: (query) => {
      const ready = query.state.data?.embedReady;
      return ready ? 60_000 : 3_000;
    },
    // Keep stale data visible while revalidating so the UI never flickers.
    staleTime: 2_000,
  });
}

export function useWorkspaceAnalyticsQuery(workspaceId: string) {
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: ["workspace-analytics", workspaceId],
    queryFn: () => apiGet<WorkspaceAnalytics>(`/workspaces/${workspaceId}/analytics`),
    enabled: Boolean(workspaceId),
    staleTime: 60_000,
    retry: 1,
  });

  async function refreshAnalytics() {
    const data = await apiGet<WorkspaceAnalytics>(
      `/workspaces/${workspaceId}/analytics?refresh=true`,
    );
    queryClient.setQueryData(["workspace-analytics", workspaceId], data);
    return data;
  }

  return { ...query, refreshAnalytics };
}
