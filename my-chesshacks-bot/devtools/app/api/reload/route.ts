import { NextResponse } from "next/server";
import { ensureServerRunning, restartServer } from "../server";

export const runtime = "nodejs";

export async function POST() {
  restartServer("manual");
  await ensureServerRunning();
  return NextResponse.json({ status: "reloaded" });
}
