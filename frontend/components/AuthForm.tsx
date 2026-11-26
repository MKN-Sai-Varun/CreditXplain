"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { homeFor, useAuth } from "./AuthProvider";
import { Alert, Button, Card, Field, Input } from "./ui";
import { errorMessage } from "@/lib/api";

export function AuthForm({ mode }: { mode: "login" | "register" }) {
  const { login, register } = useAuth();
  const router = useRouter();
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (mode === "register" && password.length < 6) {
      setError("Password must be at least 6 characters");
      return;
    }
    setBusy(true);
    try {
      const user = mode === "login" ? await login(email, password) : await register(fullName, email, password);
      router.replace(homeFor(user));
    } catch (err) {
      setError(mode === "login" ? "Email or password is incorrect" : errorMessage(err));
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-sm">
      <Card title={mode === "login" ? "Sign in" : "Create an account"}>
        <form onSubmit={submit} className="space-y-4">
          {mode === "register" && (
            <Field label="Full name">
              <Input value={fullName} onChange={(e) => setFullName(e.target.value)} required autoComplete="name" />
            </Field>
          )}
          <Field label="Email">
            <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" />
          </Field>
          <Field label="Password" hint={mode === "register" ? "At least 6 characters" : undefined}>
            <Input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete={mode === "login" ? "current-password" : "new-password"}
            />
          </Field>
          {error && <Alert>{error}</Alert>}
          <Button type="submit" disabled={busy} className="w-full">
            {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
          </Button>
        </form>
        <p className="mt-4 text-center text-sm text-slate-600">
          {mode === "login" ? (
            <>No account yet? <Link href="/register" className="font-medium text-slate-900 underline">Create one</Link></>
          ) : (
            <>Already registered? <Link href="/login" className="font-medium text-slate-900 underline">Sign in</Link></>
          )}
        </p>
      </Card>
    </div>
  );
}
