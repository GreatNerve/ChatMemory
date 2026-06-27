"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/cn";

interface AppShellProps {
  children: React.ReactNode;
  workspaceId?: string;
  workspaceName?: string;
}

function NavLink({
  href,
  label,
  active,
}: {
  href: string;
  label: string;
  active: boolean;
}) {
  return (
    <Link
      href={href}
      className={cn(
        "block border-2 px-3 py-2 font-mono text-xs uppercase tracking-widest transition-colors",
        active
          ? "border-[var(--cm-accent)] bg-[var(--cm-accent)] text-[var(--cm-accent-fg)]"
          : "border-transparent text-[var(--cm-text-muted)] hover:border-[var(--cm-border-muted)] hover:text-[var(--cm-text)]",
      )}
    >
      {label}
    </Link>
  );
}

export function AppShell({ children, workspaceId, workspaceName }: AppShellProps) {
  const pathname = usePathname();
  const base = workspaceId ? `/workspace/${workspaceId}` : "";

  return (
    <div className="flex min-h-screen flex-col">
      <header className="flex items-center justify-between border-b-2 border-[var(--cm-border)] bg-[var(--cm-surface)] px-4 py-3">
        <div className="flex items-center gap-4">
          <Link
            href="/"
            className="font-mono text-lg font-bold uppercase tracking-tight text-[var(--cm-text)]"
          >
            ChatMemory
          </Link>
          {workspaceName ? (
            <span className="font-mono text-xs uppercase tracking-widest text-[var(--cm-text-muted)]">
              / {workspaceName}
            </span>
          ) : null}
        </div>
        <Link
          href="/settings"
          className="font-mono text-xs uppercase tracking-widest text-[var(--cm-text-muted)] hover:text-[var(--cm-accent)]"
        >
          Settings
        </Link>
      </header>

      <div className="flex flex-1">
        <aside className="hidden w-56 shrink-0 border-r-4 border-[var(--cm-border)] bg-[var(--cm-surface)] p-3 md:block">
          <p className="mb-2 font-mono text-[10px] uppercase tracking-widest text-[var(--cm-text-muted)]">
            Workspaces
          </p>
          <NavLink href="/" label="All" active={pathname === "/"} />

          {workspaceId ? (
            <div className="mt-4 flex flex-col gap-1">
              <p className="mb-1 font-mono text-[10px] uppercase tracking-widest text-[var(--cm-text-muted)]">
                Active
              </p>
              <NavLink href={base} label="Overview" active={pathname === base} />
              <NavLink href={`${base}/ask`} label="Ask" active={pathname === `${base}/ask`} />
              <NavLink
                href={`${base}/people`}
                label="People"
                active={pathname.startsWith(`${base}/people`)}
              />
            </div>
          ) : null}
        </aside>

        <main className="flex-1 p-4 md:p-6">
          <div className="mx-auto max-w-6xl">{children}</div>
        </main>
      </div>
    </div>
  );
}
