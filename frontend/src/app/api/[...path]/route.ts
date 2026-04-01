import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const SHORT_TIMEOUT_MS = 12_000;
const LONG_TIMEOUT_MS = 45_000;
const ANALYZE_RETRIES = 2;

const LONG_PATHS = new Set(["analyze-incident", "analysis-feedback"]);

type RouteContext = {
  params: {
    path: string[];
  };
};

function bodyBaseUrl(): string {
  const configured =
    process.env.BODY_API_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://localhost:8001";
  return configured.replace(/\/$/, "");
}

function targetUrl(path: string[], search: string): string {
  const joinedPath = path.map(encodeURIComponent).join("/");
  return `${bodyBaseUrl()}/api/${joinedPath}${search}`;
}

function requestTimeoutMs(path: string[]): number {
  const leaf = path[path.length - 1] || "";
  if (LONG_PATHS.has(leaf)) {
    return LONG_TIMEOUT_MS;
  }
  return SHORT_TIMEOUT_MS;
}

function buildForwardHeaders(request: NextRequest): Headers {
  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  const accept = request.headers.get("accept");
  const apiKey = request.headers.get("x-api-key");
  const tenantId = request.headers.get("x-tenant-id") || "default-tenant";

  if (contentType) {
    headers.set("content-type", contentType);
  }
  if (accept) {
    headers.set("accept", accept);
  }
  if (apiKey) {
    headers.set("x-api-key", apiKey);
  }
  headers.set("x-tenant-id", tenantId);
  return headers;
}

function copyResponseHeaders(upstream: Response): Headers {
  const headers = new Headers(upstream.headers);
  headers.set("cache-control", "no-store, max-age=0");
  return headers;
}

async function proxyRequest(request: NextRequest, path: string[]): Promise<NextResponse> {
  if (path.length === 0) {
    return NextResponse.json({ detail: "missing api path" }, { status: 400 });
  }

  const method = request.method.toUpperCase();
  const body =
    method === "GET" || method === "HEAD" || method === "OPTIONS"
      ? undefined
      : await request.arrayBuffer();
  const headers = buildForwardHeaders(request);
  const url = targetUrl(path, request.nextUrl.search);
  const timeoutMs = requestTimeoutMs(path);
  const attempts = path[path.length - 1] === "analyze-incident" ? ANALYZE_RETRIES : 1;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const timeoutController = new AbortController();
    const timeoutHandle = setTimeout(() => {
      timeoutController.abort("timeout");
    }, timeoutMs);

    try {
      const upstream = await fetch(url, {
        method,
        headers,
        body,
        signal: timeoutController.signal,
        cache: "no-store",
        redirect: "manual",
      });

      return new NextResponse(upstream.body, {
        status: upstream.status,
        headers: copyResponseHeaders(upstream),
      });
    } catch (error) {
      if (attempt === attempts) {
        return NextResponse.json(
          {
            detail: "Upstream body API unavailable.",
            endpoint: path.join("/"),
            reason: error instanceof Error ? error.message : "unknown_error",
          },
          { status: 502 }
        );
      }
      await new Promise((resolve) => setTimeout(resolve, 250));
    } finally {
      clearTimeout(timeoutHandle);
    }
  }

  return NextResponse.json({ detail: "Proxy request exhausted retries." }, { status: 502 });
}

export async function GET(request: NextRequest, context: RouteContext): Promise<NextResponse> {
  return proxyRequest(request, context.params.path);
}

export async function POST(request: NextRequest, context: RouteContext): Promise<NextResponse> {
  return proxyRequest(request, context.params.path);
}

export async function PUT(request: NextRequest, context: RouteContext): Promise<NextResponse> {
  return proxyRequest(request, context.params.path);
}

export async function PATCH(request: NextRequest, context: RouteContext): Promise<NextResponse> {
  return proxyRequest(request, context.params.path);
}

export async function DELETE(request: NextRequest, context: RouteContext): Promise<NextResponse> {
  return proxyRequest(request, context.params.path);
}

export async function OPTIONS(request: NextRequest, context: RouteContext): Promise<NextResponse> {
  return proxyRequest(request, context.params.path);
}
