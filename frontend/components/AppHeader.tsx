"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";

import { homeFor, useAuth } from "./AuthProvider";
import { Button } from "./ui";

export function AppHeader() {
  const { user, logout } = useAuth();
  const router = useRouter();

  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-4">
        <Link href={user ? homeFor(user) : "/"} className="text-lg font-semibold text-slate-900">
          CreditXplain
        </Link>
        <nav className="flex items-center gap-2 text-sm">
          {user?.role === "applicant" && (
            <>
              <Link href="/applications" className="rounded-md px-3 py-2 text-slate-700 hover:bg-slate-100">
                My applications
              </Link>
              <Link href="/applications/new" className="rounded-md px-3 py-2 text-slate-700 hover:bg-slate-100">
                New application
              </Link>
            </>
          )}
          {user ? (
            <>
              <span className="hidden px-2 text-slate-500 sm:inline">{user.full_name || user.email}</span>
              <Button
                variant="secondary"
                onClick={() => {
                  logout();
                  router.replace("/login");
                }}
              >
                Sign out
              </Button>
            </>
          ) : (
            <>
              <Link href="/login" className="rounded-md px-3 py-2 text-slate-700 hover:bg-slate-100">
                Sign in
              </Link>
              <Link href="/register" className="rounded-md bg-slate-900 px-3 py-2 font-medium text-white hover:bg-slate-700">
                Create account
              </Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
