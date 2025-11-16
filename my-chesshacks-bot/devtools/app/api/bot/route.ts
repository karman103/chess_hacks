import { NextRequest, NextResponse } from "next/server";
import { ensureServerRunning } from "../server";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  console.log("/api/bot POST received");
  try {
    await ensureServerRunning();
  } catch (err) {
    console.error("server not started");
    return NextResponse.json({ error: "server not started" }, { status: 503 });
  }

  let body: any;
  try {
    body = await req.json();
  } catch (err) {
    console.error("/api/bot invalid JSON", err);
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  console.log("Proxying request to serve.py /move");

  const servePort = process.env.SERVE_PORT || "5058";

  const response = await fetch(`http://127.0.0.1:${servePort}/move`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });

  const text = await response.text();
  console.log("Received response from serve.py /move", {
    status: response.status
  });
  const contentType =
    response.headers.get("content-type") || "application/json";

  return new NextResponse(text, {
    status: response.status,
    headers: {
      "Content-Type": contentType
    }
  });
}
