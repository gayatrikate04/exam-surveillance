"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Loader2, Mail, Lock } from "lucide-react";
import {
  AuthFooterLink,
  AuthFormCard,
  AuthPageShell,
} from "@/components/auth-layout";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

function FieldIcon({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <span
      className={cn(
        "pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground",
        className
      )}
    >
      {children}
    </span>
  );
}

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = (await res.json()) as { error?: string };
      if (!res.ok) {
        setError(data.error ?? "Registration failed.");
        return;
      }
      router.push("/");
      router.refresh();
    } catch {
      setError("Could not reach the server. Check your connection and try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthPageShell
      footer={
        <span>
          Secure local sign-up ·{" "}
          <span className="text-foreground/70">Agesight Exam Activity Monitor</span>
        </span>
      }
    >
      <AuthFormCard
        title="Create your account"
        description="Set up access to your monitoring dashboard. Credentials are stored locally on the server."
      >
        <form onSubmit={onSubmit} className="space-y-6">
          {error ? (
            <Alert variant="destructive" className="border-destructive/30 bg-destructive/5">
              <AlertDescription className="text-sm">{error}</AlertDescription>
            </Alert>
          ) : null}

          <div className="space-y-5">
            <div className="space-y-2">
              <Label htmlFor="email" className="text-sm font-medium">
                Email address
              </Label>
              <div className="relative">
                <FieldIcon>
                  <Mail className="size-4" strokeWidth={1.75} />
                </FieldIcon>
                <Input
                  id="email"
                  type="email"
                  autoComplete="email"
                  placeholder="you@institution.edu"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="h-11 pl-10 text-base md:text-sm"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="password" className="text-sm font-medium">
                Password
              </Label>
              <div className="relative">
                <FieldIcon>
                  <Lock className="size-4" strokeWidth={1.75} />
                </FieldIcon>
                <Input
                  id="password"
                  type="password"
                  autoComplete="new-password"
                  placeholder="At least 6 characters"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={6}
                  className="h-11 pl-10 text-base md:text-sm"
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Use 6 or more characters. Avoid reusing passwords from other services.
              </p>
            </div>
          </div>

          <Button type="submit" size="lg" className="h-11 w-full text-sm font-medium" disabled={loading}>
            {loading ? (
              <>
                <Loader2 className="size-4 animate-spin" />
                Creating account…
              </>
            ) : (
              "Create account"
            )}
          </Button>

          <AuthFooterLink prompt="Already have an account?" href="/login" linkLabel="Sign in" />
        </form>
      </AuthFormCard>
    </AuthPageShell>
  );
}
