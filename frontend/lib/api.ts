import axios, { AxiosError } from "axios";

import type {
  ApplicationCreate,
  ApplicationDetail,
  ApplicationSummary,
  Decision,
  DocumentInfo,
  DocumentsOverview,
  RecourseOption,
  RecoursePreferences,
  TokenResponse,
  User,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const TOKEN_KEY = "creditxplain_token";

export const tokenStore = {
  get: () => (typeof window === "undefined" ? null : window.localStorage.getItem(TOKEN_KEY)),
  set: (token: string) => window.localStorage.setItem(TOKEN_KEY, token),
  clear: () => window.localStorage.removeItem(TOKEN_KEY),
};

const api = axios.create({ baseURL: API_BASE_URL });

api.interceptors.request.use((config) => {
  const token = tokenStore.get();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

/** Readable message from a FastAPI error (string detail or validation list). */
export function errorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    const detail = (err.response?.data as { detail?: unknown } | undefined)?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d: { msg?: string; loc?: (string | number)[] }) => {
          const field = d.loc?.filter((p) => p !== "body").join(".");
          return field ? `${field}: ${d.msg}` : d.msg;
        })
        .join("; ");
    }
    if (!err.response) return "Cannot reach the server. Is the backend running?";
    return `Request failed (${err.response.status})`;
  }
  return err instanceof Error ? err.message : "Something went wrong";
}

async function download(path: string, filename: string) {
  const res = await api.get(path, { responseType: "blob" });
  const url = URL.createObjectURL(res.data as Blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export const authAPI = {
  login: async (email: string, password: string): Promise<TokenResponse> => {
    const form = new URLSearchParams({ username: email, password });
    return (await api.post<TokenResponse>("/api/auth/login", form)).data;
  },
  register: async (email: string, password: string, fullName: string): Promise<User> =>
    (await api.post<User>("/api/auth/register", { email, password, full_name: fullName })).data,
  me: async (): Promise<User> => (await api.get<User>("/api/auth/me")).data,
};

export const applicationAPI = {
  list: async (): Promise<ApplicationSummary[]> =>
    (await api.get<ApplicationSummary[]>("/api/applications/")).data,
  create: async (body: ApplicationCreate): Promise<ApplicationSummary> =>
    (await api.post<ApplicationSummary>("/api/applications/", body)).data,
  detail: async (id: number): Promise<ApplicationDetail> =>
    (await api.get<ApplicationDetail>(`/api/applications/${id}`)).data,
  predict: async (id: number): Promise<Decision> =>
    (await api.post<Decision>(`/api/predictions/${id}`)).data,
  recourse: async (id: number, prefs: RecoursePreferences): Promise<RecourseOption[]> =>
    (await api.post<RecourseOption[]>(`/api/predictions/${id}/recourse`, prefs)).data,
  downloadReport: (id: number) => download(`/api/applications/${id}/report.pdf`, `creditxplain-application-${id}.pdf`),
};

export const documentAPI = {
  list: async (appId: number): Promise<DocumentsOverview> =>
    (await api.get<DocumentsOverview>(`/api/applications/${appId}/documents`)).data,
  upload: async (appId: number, file: File, docType: string): Promise<DocumentInfo> => {
    const form = new FormData();
    form.append("file", file);
    form.append("doc_type", docType);
    return (await api.post<DocumentInfo>(`/api/applications/${appId}/documents`, form)).data;
  },
  download: (appId: number, doc: DocumentInfo) =>
    download(`/api/applications/${appId}/documents/${doc.id}`, doc.original_name),
};

export default api;
