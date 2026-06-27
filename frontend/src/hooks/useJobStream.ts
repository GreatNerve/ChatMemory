"use client";

import { useEffect, useRef, useState } from "react";
import { apiUrl } from "@/lib/api/client";
import { JobSnapshot } from "@/lib/api/types";

/** SSE job updates — one connection, no polling. */
export function useJobStream(jobId: string | null, onDone?: () => void) {
  const [job, setJob] = useState<JobSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;
  const doneCalledRef = useRef(false);

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      setError(null);
      return;
    }

    doneCalledRef.current = false;
    setError(null);

    const es = new EventSource(apiUrl(`/jobs/${jobId}/stream`));

    function applySnapshot(data: JobSnapshot) {
      setJob(data);
      if (data.status === "done" && !doneCalledRef.current) {
        doneCalledRef.current = true;
        onDoneRef.current?.();
      }
      if (data.status === "error") {
        setError(data.error ?? data.message ?? "Job failed");
      }
    }

    es.addEventListener("progress", (ev) => {
      try {
        applySnapshot(JSON.parse((ev as MessageEvent).data) as JobSnapshot);
      } catch {
        setError("Invalid job stream payload");
      }
    });

    es.addEventListener("done", (ev) => {
      try {
        applySnapshot(JSON.parse((ev as MessageEvent).data) as JobSnapshot);
      } catch {
        setError("Invalid job stream payload");
      }
      es.close();
    });

    es.addEventListener("error", (ev) => {
      if ((ev as MessageEvent).data) {
        try {
          const payload = JSON.parse((ev as MessageEvent).data) as { message?: string };
          setError(payload.message ?? "Job failed");
        } catch {
          setError("Job failed");
        }
      } else if (es.readyState === EventSource.CLOSED) {
        return;
      } else {
        setError("Job stream disconnected — is the backend running?");
      }
      es.close();
    });

    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) return;
      setError("Job stream disconnected — is the backend running?");
      es.close();
    };

    return () => {
      es.close();
    };
  }, [jobId]);

  return { job, error };
}
