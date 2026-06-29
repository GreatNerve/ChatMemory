const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000/api/v1";

export class ApiError extends Error {
  status: number;
  body: unknown;

  constructor(status: number, body: unknown) {
    let message = `Request failed (${status})`;
    if (typeof body === "string" && body.trim()) {
      message = body;
    } else if (
      typeof body === "object" &&
      body !== null &&
      "detail" in body &&
      typeof (body as { detail: unknown }).detail === "string"
    ) {
      message = (body as { detail: string }).detail;
    }
    super(message);
    this.status = status;
    this.body = body;
  }
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    throw new ApiError(response.status, body);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(apiUrl(path), { cache: "no-store" });
  return parseResponse<T>(response);
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
  });
  return parseResponse<T>(response);
}

export async function apiPut<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseResponse<T>(response);
}

export async function apiDelete(path: string): Promise<void> {
  const response = await fetch(apiUrl(path), { method: "DELETE" });
  await parseResponse<void>(response);
}

export type PersonaStreamEvent =
  | { token: string }
  | { done: true; interactionId?: string; debugMeta?: import("./types").ChatDebugMeta }
  | { status: "thinking" }
  | { msg_break: true }
  | { error: string }
  | import("./types").StageEvent;

/** Stream persona chat tokens via SSE (POST + readable stream). */
export async function streamPersonaChat(
  workspaceId: string,
  personId: string,
  body: {
    message: string;
    history: { role: string; content: string }[];
    previousInteractionId?: string | null;
    conversationSummary?: string | null;
  },
  onEvent: (ev: PersonaStreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(
    apiUrl(`/workspaces/${workspaceId}/people/${personId}/chat/stream`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    },
  );

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const errBody = (await response.json()) as { detail?: string };
      if (errBody.detail) detail = errBody.detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(response.status, detail);
  }

  if (!response.body) {
    throw new ApiError(response.status, "No response body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      for (const line of part.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const payload = JSON.parse(line.slice(6)) as PersonaStreamEvent;
        onEvent(payload);
      }
    }
  }
}

/** Summarize older persona chat turns for rolling context compression. */
export async function summarizePersonaChat(
  workspaceId: string,
  personId: string,
  body: { history: { role: string; content: string }[]; keepRecent?: number },
): Promise<{ summary: string; summarizedTurnCount: number }> {
  return apiPost(
    `/workspaces/${workspaceId}/people/${personId}/chat/summarize`,
    body,
  );
}
