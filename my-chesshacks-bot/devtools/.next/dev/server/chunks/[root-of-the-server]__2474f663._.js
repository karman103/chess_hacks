module.exports = [
"[externals]/next/dist/compiled/next-server/app-route-turbo.runtime.dev.js [external] (next/dist/compiled/next-server/app-route-turbo.runtime.dev.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/compiled/next-server/app-route-turbo.runtime.dev.js", () => require("next/dist/compiled/next-server/app-route-turbo.runtime.dev.js"));

module.exports = mod;
}),
"[externals]/next/dist/compiled/@opentelemetry/api [external] (next/dist/compiled/@opentelemetry/api, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/compiled/@opentelemetry/api", () => require("next/dist/compiled/@opentelemetry/api"));

module.exports = mod;
}),
"[externals]/next/dist/compiled/next-server/app-page-turbo.runtime.dev.js [external] (next/dist/compiled/next-server/app-page-turbo.runtime.dev.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/compiled/next-server/app-page-turbo.runtime.dev.js", () => require("next/dist/compiled/next-server/app-page-turbo.runtime.dev.js"));

module.exports = mod;
}),
"[externals]/next/dist/server/app-render/work-unit-async-storage.external.js [external] (next/dist/server/app-render/work-unit-async-storage.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/server/app-render/work-unit-async-storage.external.js", () => require("next/dist/server/app-render/work-unit-async-storage.external.js"));

module.exports = mod;
}),
"[externals]/next/dist/server/app-render/work-async-storage.external.js [external] (next/dist/server/app-render/work-async-storage.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/server/app-render/work-async-storage.external.js", () => require("next/dist/server/app-render/work-async-storage.external.js"));

module.exports = mod;
}),
"[externals]/next/dist/shared/lib/no-fallback-error.external.js [external] (next/dist/shared/lib/no-fallback-error.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/shared/lib/no-fallback-error.external.js", () => require("next/dist/shared/lib/no-fallback-error.external.js"));

module.exports = mod;
}),
"[externals]/next/dist/server/app-render/after-task-async-storage.external.js [external] (next/dist/server/app-render/after-task-async-storage.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/server/app-render/after-task-async-storage.external.js", () => require("next/dist/server/app-render/after-task-async-storage.external.js"));

module.exports = mod;
}),
"[externals]/child_process [external] (child_process, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("child_process", () => require("child_process"));

module.exports = mod;
}),
"[externals]/http [external] (http, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("http", () => require("http"));

module.exports = mod;
}),
"[externals]/path [external] (path, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("path", () => require("path"));

module.exports = mod;
}),
"[externals]/fs [external] (fs, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("fs", () => require("fs"));

module.exports = mod;
}),
"[project]/app/api/server.ts [app-route] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "ensureServerRunning",
    ()=>ensureServerRunning,
    "getLastRestartInfo",
    ()=>getLastRestartInfo,
    "restartServer",
    ()=>restartServer,
    "runtime",
    ()=>runtime
]);
var __TURBOPACK__imported__module__$5b$externals$5d2f$child_process__$5b$external$5d$__$28$child_process$2c$__cjs$29$__ = __turbopack_context__.i("[externals]/child_process [external] (child_process, cjs)");
var __TURBOPACK__imported__module__$5b$externals$5d2f$http__$5b$external$5d$__$28$http$2c$__cjs$29$__ = __turbopack_context__.i("[externals]/http [external] (http, cjs)");
var __TURBOPACK__imported__module__$5b$externals$5d2f$path__$5b$external$5d$__$28$path$2c$__cjs$29$__ = __turbopack_context__.i("[externals]/path [external] (path, cjs)");
var __TURBOPACK__imported__module__$5b$externals$5d2f$fs__$5b$external$5d$__$28$fs$2c$__cjs$29$__ = __turbopack_context__.i("[externals]/fs [external] (fs, cjs)");
;
;
;
;
const runtime = "nodejs";
let serverProcess = null;
let serverReadyPromise = null;
let watcher = null;
let restartTimeout = null;
let stoppingPromise = null;
let restarting = false;
let lastRestartAt = null;
let lastRestartReason = null;
const servePort = Number(process.env.SERVE_PORT || "5058");
console.log("[devtools env]", {
    PYTHON_EXECUTABLE: process.env.PYTHON_EXECUTABLE,
    PYTHON: process.env.PYTHON,
    SERVE_PORT: process.env.SERVE_PORT,
    PORT: process.env.PORT
});
function getServerCwd() {
    return __TURBOPACK__imported__module__$5b$externals$5d2f$path__$5b$external$5d$__$28$path$2c$__cjs$29$__["default"].join(process.cwd(), "..");
}
function setupWatcher() {
    if (watcher) {
        return;
    }
    const serverCwd = getServerCwd();
    const srcDir = __TURBOPACK__imported__module__$5b$externals$5d2f$path__$5b$external$5d$__$28$path$2c$__cjs$29$__["default"].join(serverCwd, "src");
    if (!__TURBOPACK__imported__module__$5b$externals$5d2f$fs__$5b$external$5d$__$28$fs$2c$__cjs$29$__["default"].existsSync(srcDir)) {
        return;
    }
    try {
        watcher = __TURBOPACK__imported__module__$5b$externals$5d2f$fs__$5b$external$5d$__$28$fs$2c$__cjs$29$__["default"].watch(srcDir, {
            recursive: true
        }, ()=>{
            console.log("Detected change in src directory, scheduling serve.py restart");
            if (restartTimeout) {
                clearTimeout(restartTimeout);
            }
            restartTimeout = setTimeout(()=>{
                restartServer("file-change");
            }, 300);
        });
    } catch (err) {
        console.error("Failed to watch src directory for changes", err);
    }
}
function stopServer() {
    if (!serverProcess) {
        serverReadyPromise = null;
        return Promise.resolve();
    }
    if (stoppingPromise) {
        return stoppingPromise;
    }
    const current = serverProcess;
    stoppingPromise = new Promise((resolve)=>{
        current.once("exit", ()=>{
            serverProcess = null;
            serverReadyPromise = null;
            stoppingPromise = null;
            resolve();
        });
        if (!current.killed) {
            current.kill();
        }
    });
    return stoppingPromise;
}
function startServer() {
    if (serverProcess && !serverProcess.killed) {
        return;
    }
    const serverCwd = getServerCwd();
    const pythonExecutable = process.env.PYTHON_EXECUTABLE || process.env.PYTHON || "python3";
    setupWatcher();
    console.log("Starting serve.py process", {
        pythonExecutable,
        serverCwd
    });
    try {
        serverProcess = (0, __TURBOPACK__imported__module__$5b$externals$5d2f$child_process__$5b$external$5d$__$28$child_process$2c$__cjs$29$__["spawn"])(pythonExecutable, [
            "-u",
            "serve.py"
        ], {
            cwd: serverCwd,
            stdio: [
                "ignore",
                "pipe",
                "pipe"
            ],
            shell: false
        });
        if (serverProcess.stdout) {
            serverProcess.stdout.on("data", (data)=>{
                const text = data.toString();
                if (text.trim().length > 0) {
                    console.log("[serve.py stdout]", text.trimEnd());
                }
            });
        }
        if (serverProcess.stderr) {
            serverProcess.stderr.on("data", (data)=>{
                const text = data.toString();
                if (text.trim().length > 0) {
                    console.error("[serve.py stderr]", text.trimEnd());
                }
            });
        }
    } catch (err) {
        console.error("Failed to spawn serve.py", err);
        serverProcess = null;
        serverReadyPromise = null;
        throw err;
    }
    serverProcess.on("exit", ()=>{
        console.log("serve.py process exited");
        serverProcess = null;
        serverReadyPromise = null;
    });
}
function waitForServerReady() {
    if (serverReadyPromise) return serverReadyPromise;
    console.log("Waiting for serve.py to become ready");
    serverReadyPromise = new Promise((resolve, reject)=>{
        const maxAttempts = 30;
        const delayMs = 200;
        let attempts = 0;
        const check = ()=>{
            console.log("Checking serve.py readiness", {
                attempts
            });
            attempts += 1;
            const req = __TURBOPACK__imported__module__$5b$externals$5d2f$http__$5b$external$5d$__$28$http$2c$__cjs$29$__["default"].request({
                host: "127.0.0.1",
                port: servePort,
                path: "/",
                method: "POST"
            }, (res)=>{
                if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
                    console.log("serve.py is ready");
                    resolve();
                } else if (attempts < maxAttempts) {
                    setTimeout(check, delayMs);
                } else {
                    reject(new Error("serve.py did not become ready"));
                }
            });
            req.on("error", (err)=>{
                console.error("Error checking serve.py readiness");
                if (attempts < maxAttempts) {
                    setTimeout(check, delayMs);
                } else {
                    reject(new Error("serve.py did not become ready"));
                }
            });
            req.end();
        };
        check();
    });
    return serverReadyPromise;
}
async function ensureServerRunning() {
    if (!serverProcess || serverProcess.killed) {
        startServer();
    }
    await waitForServerReady();
}
async function restartServer(reason) {
    if (restarting) {
        return;
    }
    restarting = true;
    console.log("Restarting serve.py process");
    lastRestartAt = Date.now();
    lastRestartReason = reason ?? null;
    await stopServer();
    startServer();
    restarting = false;
}
function getLastRestartInfo() {
    return {
        lastRestartAt,
        lastRestartReason
    };
}
}),
"[project]/app/api/bot/route.ts [app-route] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "POST",
    ()=>POST,
    "runtime",
    ()=>runtime
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/server.js [app-route] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$api$2f$server$2e$ts__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/app/api/server.ts [app-route] (ecmascript)");
;
;
const runtime = "nodejs";
async function POST(req) {
    console.log("/api/bot POST received");
    try {
        await (0, __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$api$2f$server$2e$ts__$5b$app$2d$route$5d$__$28$ecmascript$29$__["ensureServerRunning"])();
    } catch (err) {
        console.error("server not started");
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json({
            error: "server not started"
        }, {
            status: 503
        });
    }
    let body;
    try {
        body = await req.json();
    } catch (err) {
        console.error("/api/bot invalid JSON", err);
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json({
            error: "Invalid JSON body"
        }, {
            status: 400
        });
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
    const contentType = response.headers.get("content-type") || "application/json";
    return new __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"](text, {
        status: response.status,
        headers: {
            "Content-Type": contentType
        }
    });
}
}),
];

//# sourceMappingURL=%5Broot-of-the-server%5D__2474f663._.js.map