import { NextResponse } from "next/server";
import { getLastRestartInfo } from "../../server";

export const runtime = "nodejs";

export async function GET() {
  const info = getLastRestartInfo();
  return NextResponse.json(info);
}
