import Link from "next/link";
import { Eye, ShieldCheck, Sparkles } from "lucide-react";
import { APP_BRAND, APP_TITLE } from "@/lib/branding";
import { cn } from "@/lib/utils";

const FEATURES = [
  { icon: Eye, label: "Live exam monitoring" },
  { icon: ShieldCheck, label: "Activity risk detection" },
  { icon: Sparkles, label: "Image & video analysis" },
] as const;

export function AuthBrandPanel({ className }: { className?: string }) {
  return (
    <aside
      className={cn(
        "relative flex flex-col justify-between overflow-hidden px-10 py-12 text-white lg:px-14 lg:py-16",
        className
      )}
    >
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 bg-[linear-gradient(145deg,oklch(0.22_0.04_250)_0%,oklch(0.16_0.03_265)_45%,oklch(0.12_0.02_280)_100%)]"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -right-24 -top-24 size-80 rounded-full bg-white/10 blur-3xl"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -bottom-32 -left-16 size-96 rounded-full bg-white/5 blur-3xl"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.35] [background-image:radial-gradient(circle_at_1px_1px,rgba(255,255,255,0.12)_1px,transparent_0)] [background-size:24px_24px]"
      />

      <div className="relative z-10">
        <div className="mb-8 flex size-12 items-center justify-center rounded-2xl bg-white/15 ring-1 ring-white/20 backdrop-blur-sm">
          <span className="font-heading text-xl font-semibold tracking-tight">A</span>
        </div>
        <h1 className="font-heading text-4xl font-semibold tracking-tight lg:text-5xl">{APP_BRAND}</h1>
        <p className="mt-3 max-w-sm text-base leading-relaxed text-white/75">
          Exam activity monitoring for classrooms and proctored sessions.
        </p>
      </div>

      <ul className="relative z-10 mt-12 hidden space-y-4 sm:block">
        {FEATURES.map(({ icon: Icon, label }) => (
          <li key={label} className="flex items-center gap-3 text-sm text-white/85">
            <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-white/10 ring-1 ring-white/15">
              <Icon className="size-4" strokeWidth={1.75} />
            </span>
            {label}
          </li>
        ))}
      </ul>

      <p className="relative z-10 mt-10 text-xs text-white/50">{APP_TITLE}</p>
    </aside>
  );
}

export function AuthPageShell({
  children,
  footer,
}: {
  children: React.ReactNode;
  footer?: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-background">
      <div className="grid min-h-screen lg:grid-cols-[1.05fr_1fr]">
        <AuthBrandPanel className="hidden lg:flex" />

        <main className="flex flex-col">
          <div className="border-b px-6 py-8 lg:hidden">
            <p className="font-heading text-2xl font-semibold tracking-tight">{APP_BRAND}</p>
            <p className="mt-1 text-sm text-muted-foreground">Exam activity monitor</p>
          </div>

          <div className="flex flex-1 flex-col items-center justify-center px-6 py-10 sm:px-10">
            <div className="w-full max-w-[400px]">{children}</div>
          </div>

          {footer ? (
            <footer className="border-t px-6 py-4 text-center text-xs text-muted-foreground">
              {footer}
            </footer>
          ) : null}
        </main>
      </div>
    </div>
  );
}

export function AuthFormCard({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <h2 className="font-heading text-2xl font-semibold tracking-tight text-foreground">{title}</h2>
        {description ? <p className="text-sm leading-relaxed text-muted-foreground">{description}</p> : null}
      </header>
      {children}
    </div>
  );
}

export function AuthFooterLink({
  prompt,
  href,
  linkLabel,
}: {
  prompt: string;
  href: string;
  linkLabel: string;
}) {
  return (
    <p className="text-center text-sm text-muted-foreground">
      {prompt}{" "}
      <Link
        href={href}
        className="font-medium text-foreground underline-offset-4 transition-colors hover:text-primary hover:underline"
      >
        {linkLabel}
      </Link>
    </p>
  );
}
