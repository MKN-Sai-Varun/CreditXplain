"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { homeFor, useAuth } from "@/components/AuthProvider";
import { Spinner } from "@/components/ui";

export default function Home() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (user) router.replace(homeFor(user));
  }, [user, router]);

  if (loading || user) return <Spinner />;

  return (
    <div className="mx-auto max-w-2xl py-12 text-center">
      <h1 className="text-3xl font-semibold text-slate-900">Credit decisions you can understand</h1>
      <p className="mt-4 text-slate-600">
        Apply for a loan and see the decision, the main reasons behind it, and concrete steps that could
        improve it, such as a smaller loan, a longer repayment period or re-applying later.
      </p>
      <div className="mt-8 flex justify-center gap-3">
        <Link href="/register" className="rounded-md bg-slate-900 px-5 py-2.5 text-sm font-medium text-white hover:bg-slate-700">
          Create an account
        </Link>
        <Link href="/login" className="rounded-md border border-slate-300 bg-white px-5 py-2.5 text-sm font-medium text-slate-800 hover:bg-slate-50">
          Sign in
        </Link>
      </div>
    </div>
  );
}
