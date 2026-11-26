"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { homeFor, useAuth } from "./AuthProvider";
import { Spinner } from "./ui";
import type { Role } from "@/lib/types";

/** Shows children only to a signed-in user with an allowed role; sends everyone else away. */
export function RequireAuth({ roles, children }: { roles: Role[]; children: React.ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();
  const allowed = user != null && roles.includes(user.role);

  useEffect(() => {
    if (loading) return;
    if (!user) router.replace("/login");
    else if (!roles.includes(user.role)) router.replace(homeFor(user));
  }, [loading, user, roles, router]);

  if (loading || !allowed) return <Spinner />;
  return <>{children}</>;
}
