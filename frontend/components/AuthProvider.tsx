"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { authAPI, tokenStore } from "@/lib/api";
import type { User } from "@/lib/types";

interface AuthState {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<User>;
  register: (fullName: string, email: string, password: string) => Promise<User>;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // restore the session saved in this browser, if any
  useEffect(() => {
    let cancelled = false;
    async function restore() {
      const me = await (tokenStore.get()
        ? authAPI.me().catch(() => {
            tokenStore.clear();   // expired or invalid token
            return null;
          })
        : Promise.resolve(null));
      if (!cancelled) {
        setUser(me);
        setLoading(false);
      }
    }
    restore();
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const res = await authAPI.login(email, password);
    tokenStore.set(res.access_token);
    setUser(res.user);
    return res.user;
  }, []);

  const register = useCallback(
    async (fullName: string, email: string, password: string) => {
      await authAPI.register(email, password, fullName);
      return login(email, password);
    },
    [login],
  );

  const logout = useCallback(() => {
    tokenStore.clear();
    setUser(null);
  }, []);

  const value = useMemo(() => ({ user, loading, login, register, logout }), [user, loading, login, register, logout]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}

/** Where a signed-in user lands. */
export const homeFor = (user: User) => (user.role === "applicant" ? "/applications" : "/officer");
