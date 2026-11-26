"use client";

import { useState } from "react";

import { Alert, Button, Card, Field, Select } from "../ui";
import { documentAPI, errorMessage } from "@/lib/api";
import { date } from "@/lib/format";
import type { DocumentInfo, DocumentRequest } from "@/lib/types";

const DOC_TYPES: Record<string, string> = {
  income_proof: "Income proof",
  id_proof: "ID proof",
  address_proof: "Address proof",
  bank_statement: "Bank statement",
  employment_proof: "Employment proof",
  other: "Other",
};
const MAX_BYTES = 5 * 1024 * 1024;

const size = (n: number) => (n < 1024 ? `${n} bytes` : n < 1024 * 1024 ? `${Math.round(n / 1024)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`);

export function Documents({
  appId,
  initialDocuments,
  requests,
}: {
  appId: number;
  initialDocuments: DocumentInfo[];
  requests: DocumentRequest[];
}) {
  const [docs, setDocs] = useState(initialDocuments);
  const [file, setFile] = useState<File | null>(null);
  const [docType, setDocType] = useState("income_proof");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inputKey, setInputKey] = useState(0);

  async function upload(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setError(null);
    if (file.size > MAX_BYTES) {
      setError("The file is larger than 5 MB.");
      return;
    }
    setBusy(true);
    try {
      const doc = await documentAPI.upload(appId, file, docType);
      setDocs((prev) => [...prev, doc]);
      setFile(null);
      setInputKey((k) => k + 1);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="Documents">
      {requests.length > 0 && (
        <Alert kind="info">
          <p className="font-medium">A loan officer has asked for:</p>
          <ul className="mt-1 list-disc pl-5">
            {requests.map((r) => (
              <li key={r.requested_at}>
                {r.message} <span className="text-slate-500">({date(r.requested_at)})</span>
              </li>
            ))}
          </ul>
        </Alert>
      )}
      {docs.length > 0 && (
        <ul className="mt-3 divide-y divide-slate-100 text-sm">
          {docs.map((d) => (
            <li key={d.id} className="flex items-center justify-between py-2">
              <span>
                {DOC_TYPES[d.doc_type] ?? d.doc_type} · {d.original_name}{" "}
                <span className="text-slate-500">({size(d.size_bytes)}, {date(d.uploaded_at)})</span>
              </span>
              <Button variant="ghost" onClick={() => documentAPI.download(appId, d).catch((err) => setError(errorMessage(err)))}>
                Download
              </Button>
            </li>
          ))}
        </ul>
      )}
      <form onSubmit={upload} className="mt-4 grid items-end gap-3 sm:grid-cols-[1fr_1fr_auto]">
        <Field label="Document type">
          <Select
            options={Object.values(DOC_TYPES)}
            value={DOC_TYPES[docType]}
            onChange={(e) => setDocType(Object.keys(DOC_TYPES).find((k) => DOC_TYPES[k] === e.target.value) ?? "other")}
          />
        </Field>
        <Field label="File" hint="PDF, PNG or JPEG, up to 5 MB">
          <input
            key={inputKey}
            type="file"
            accept="application/pdf,image/png,image/jpeg"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="block w-full text-sm text-slate-700 file:mr-3 file:rounded-md file:border-0 file:bg-slate-100 file:px-3 file:py-2 file:text-sm file:font-medium"
          />
        </Field>
        <Button type="submit" disabled={!file || busy}>
          {busy ? "Uploading…" : "Upload"}
        </Button>
      </form>
      {error && (
        <div className="mt-3">
          <Alert>{error}</Alert>
        </div>
      )}
    </Card>
  );
}
