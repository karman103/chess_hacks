"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import Script from "next/script";

export default function AnalysisBoardWrapper() {
  const [scriptsLoaded, setScriptsLoaded] = useState({
    jquery: false,
    analysis: false
  });
  const boardContainerRef = useRef<HTMLDivElement>(null);
  const chessInstanceRef = useRef<any>(null);
  const [boardInitialized, setBoardInitialized] = useState(false);
  const [reloading, setReloading] = useState(false);
  const [engineError, setEngineError] = useState<string | null>(null);
  const [reloadStatus, setReloadStatus] = useState<string | null>(null);
  const lastRestartAtRef = useRef<number | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const hasJQuery = Boolean((window as any).jQuery || (window as any).$);
    const hasInit = typeof (window as any).initAnalysisBoard === "function";

    if (!scriptsLoaded.jquery && hasJQuery) {
      setScriptsLoaded((prev) => ({ ...prev, jquery: true }));
    }

    if (!scriptsLoaded.analysis && hasInit) {
      setScriptsLoaded((prev) => ({ ...prev, analysis: true }));
    }
  }, [scriptsLoaded.jquery, scriptsLoaded.analysis]);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const handler = (event: ErrorEvent) => {
      const message = event.message || String((event as any).error || "");
      const filename = event.filename || "";
      const isEngineFile = filename.includes(
        "/analysis-board/src/engines/stockfish"
      );
      const mentionsSharedArrayBuffer = message.includes("SharedArrayBuffer");
      const mentionsStockfish = message.toLowerCase().includes("stockfish");

      if (isEngineFile || mentionsSharedArrayBuffer || mentionsStockfish) {
        setEngineError(
          "Engine failed to load in this browser. Some Stockfish versions may not be supported."
        );
      }
    };

    window.addEventListener("error", handler);
    return () => {
      window.removeEventListener("error", handler);
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;

    let cancelled = false;

    const fetchStatus = async () => {
      try {
        const res = await fetch("/api/reload/status");
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        if (typeof data.lastRestartAt === "number") {
          lastRestartAtRef.current = data.lastRestartAt;
        }
      } catch {}
    };

    fetchStatus();

    const interval = setInterval(async () => {
      try {
        const res = await fetch("/api/reload/status");
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        if (typeof data.lastRestartAt === "number") {
          const prev = lastRestartAtRef.current;
          const current = data.lastRestartAt as number;
          if (!prev || current > prev) {
            lastRestartAtRef.current = current;
            const reason = data.lastRestartReason as string | null | undefined;
            if (reason === "manual") {
              setReloadStatus("Server reloaded");
            } else if (reason === "file-change") {
              setReloadStatus("Server reloaded after file change");
            } else {
              setReloadStatus("Server reloaded");
            }
            setTimeout(() => {
              setReloadStatus(null);
            }, 3000);
          }
        }
      } catch {}
    }, 3000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const initializeAnalysisBoard = useCallback(async () => {
    try {
      const gameDataForAnalysis = {
        pgn: "",
        white: {
          name: "You",
          elo: "N/A"
        },
        black: {
          name: "Bot",
          elo: "N/A"
        },
        mode: "play-vs-bot",
        timeleft: 60000,
        playerColor: "white"
      };

      console.log("AnalysisBoardWrapper: initializing board", {
        playerColor: gameDataForAnalysis.playerColor
      });

      (window as any).gameDataForAnalysis = gameDataForAnalysis;

      // wait on init function available
      if (typeof (window as any).initAnalysisBoard === "function") {
        if (chessInstanceRef.current?.destroy) {
          chessInstanceRef.current.destroy();
          chessInstanceRef.current = null;
        }
        const instance = await (window as any).initAnalysisBoard(
          gameDataForAnalysis
        );
        chessInstanceRef.current = instance;
      } else {
        console.error("initAnalysisBoard function not found on window");
      }
    } catch (error) {
      console.error("Failed to initialize analysis board:", error);
    }
  }, []);

  useEffect(() => {
    // init
    if (scriptsLoaded.jquery && scriptsLoaded.analysis && !boardInitialized) {
      setBoardInitialized(true);
      initializeAnalysisBoard();
    }
  }, [
    scriptsLoaded.jquery,
    scriptsLoaded.analysis,
    boardInitialized,
    initializeAnalysisBoard
  ]);

  useEffect(() => {
    return () => {
      if (chessInstanceRef.current?.destroy) {
        chessInstanceRef.current.destroy();
        chessInstanceRef.current = null;
      }
    };
  }, []);

  const handleReloadClick = useCallback(async () => {
    if (reloading) return;
    setReloading(true);
    console.log("User pressed manual reload button");
    try {
      await fetch("/api/reload", {
        method: "POST"
      });
    } catch (error) {
      console.error("Failed to reload Python server", error);
    } finally {
      setReloading(false);
    }
  }, [reloading]);

  return (
    <div className="h-full w-full grid place-items-center">
      <div>
        <div className="text-center mb-4">
          <p className="font-bold">ChessHacks Devtools</p>
          <p>
            See <span className="text-sm p-1 rounded font-mono">README.md</span>{" "}
            for further instructions
          </p>
        </div>
        {engineError && (
          <div className="mb-4 rounded border border-red-500 bg-red-950 px-3 py-2 text-sm text-red-200">
            {engineError}
          </div>
        )}
        {/* board scoped css */}
        <link
          rel="stylesheet"
          href="/analysis-board/css/style.css"
          type="text/css"
        />

        {/* we need jquery */}
        <Script
          src="/analysis-board/libs/jquery.3.7.1.min.js"
          strategy="afterInteractive"
          onLoad={() => {
            setScriptsLoaded((prev) => ({ ...prev, jquery: true }));
          }}
        />

        {/* chess.js is loaded automatically as a module dependency */}
        <Script
          src="/analysis-board/init-game.js"
          type="module"
          strategy="afterInteractive"
          onLoad={() => {
            setScriptsLoaded((prev) => ({ ...prev, analysis: true }));
          }}
        />

        <div
          ref={boardContainerRef}
          className="analysis-board-container dark_theme"
        >
          <div className="mb-4 flex gap-2 items-center min-h-[1.5rem]"></div>
          <button
            className="underline cursor-pointer disabled:opacity-50"
            disabled={reloading}
            onClick={handleReloadClick}
          >
            {reloading ? "Reloading /src..." : "Reload /src"}
          </button>
          {reloadStatus && (
            <span className="ml-2 text-xs text-green-400">{reloadStatus}</span>
          )}

          <main>
            <article className="container">
              <div className="main-panel">
                <div className="chess-container">
                  <div className="nameplate top">
                    <div className="profile">
                      <h4 id="black-name" className="name">
                        Black
                      </h4>
                    </div>
                    <div className="clock" id="black-clock">
                      <span className="clock-time">--:--</span>
                    </div>
                  </div>

                  <div className="chess-box">
                    <div className="eval-bar-container">
                      <div className="eval-bar">
                        <div className="eval-text"></div>
                        <div className="eval-fill"></div>
                      </div>
                    </div>
                    <div id="chessboard"></div>
                  </div>

                  <div className="nameplate bottom">
                    <div className="profile">
                      <h4 id="white-name" className="name">
                        White
                      </h4>
                    </div>
                    <div className="clock" id="white-clock">
                      <span className="clock-time">--:--</span>
                    </div>
                  </div>
                </div>

                <div className="sidebar">
                  <div className="sidebar-header">
                    <div className="tab-buttons">
                      <button className="tab-button active" data-tab="moves">
                        Moves
                      </button>
                      <button className="tab-button" data-tab="debug">
                        Debug
                      </button>
                      <button className="tab-button" data-tab="settings">
                        Settings
                      </button>
                    </div>
                    <div className="evaluation-progress-container">
                      <div className="evaluation-progress-bar">
                        <div className="progress-bar-fill"></div>
                      </div>
                    </div>
                  </div>
                  <div className="tab-content blur-content">
                    <div id="moves-tab" className="tab-panel active">
                      <div className="top-content move-info"></div>
                      <div className="top-content engine-lines"></div>
                      <div className="moves-container">
                        <div id="move-tree" className="move-tree"></div>
                      </div>
                    </div>
                    <div id="debug-tab" className="tab-panel">
                      <div className="debug-probabilities"></div>
                      <div className="debug-logs-section"></div>
                    </div>
                    <div id="settings-tab" className="tab-panel">
                      <div className="settings-menu-container"></div>
                    </div>
                  </div>

                  <div className="bottom-content blur-content">
                    <div className="controls">
                      <button id="restart">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 320 512"
                        >
                          <path
                            d="M267.5 440.6c9.5 7.9 22.8 9.7 34.1 4.4s18.4-16.6 18.4-29l0-320c0-12.4-7.2-23.7-18.4-29s-24.5-3.6-34.1 4.4l-192 160L64 241 64 96c0-17.7-14.3-32-32-32S0 78.3 0 96L0 416c0 17.7 14.3 32 32 32s32-14.3 32-32l0-145 11.5 9.6 192 160z"
                            fill="currentColor"
                          />
                        </svg>
                      </button>
                      <button id="backward">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 448 512"
                        >
                          <path
                            d="M9.4 233.4c-12.5 12.5-12.5 32.8 0 45.3l160 160c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3L109.2 288 416 288c17.7 0 32-14.3 32-32s-14.3-32-32-32l-306.7 0L214.6 118.6c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0l-160 160z"
                            fill="currentColor"
                          />
                        </svg>
                      </button>
                      <div className="quick-menu-container">
                        <button id="popup-quick-menu">
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 448 512"
                          >
                            <path
                              d="M0 96C0 78.3 14.3 64 32 64l384 0c17.7 0 32 14.3 32 32s-14.3 32-32 32L32 128C14.3 128 0 113.7 0 96zM0 256c0-17.7 14.3-32 32-32l384 0c17.7 0 32 14.3 32 32s-14.3 32-32 32L32 288c-17.7 0-32-14.3-32-32zM448 416c0 17.7-14.3 32-32 32L32 448c-17.7 0-32-14.3-32-32s14.3-32 32-32l384 0c17.7 0 32 14.3 32 32z"
                              fill="currentColor"
                            />
                          </svg>
                        </button>
                        <div className="quick-menu" id="quick-menu">
                          <div className="quick-menu-item" id="flip-board">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              viewBox="0 0 512 512"
                            >
                              <path
                                d="M105.1 202.6c7.7-21.8 20.2-42.3 37.8-59.8c62.5-62.5 163.8-62.5 226.3 0L386.3 160H352c-17.7 0-32 14.3-32 32s14.3 32 32 32H463.5c0 0 0 0 0 0h.4c17.7 0 32-14.3 32-32V80c0-17.7-14.3-32-32-32s-32 14.3-32 32v35.2L414.4 97.6c-87.5-87.5-229.3-87.5-316.8 0C73.2 122 55.6 150.7 44.8 181.4c-5.9 16.7 2.9 34.9 19.5 40.8s34.9-2.9 40.8-19.5zM39 289.3c-5 1.5-9.8 4.2-13.7 8.2c-4 4-6.7 8.8-8.1 14c-.3 1.2-.6 2.5-.8 3.8c-.3 1.7-.4 3.4-.4 5.1V448c0 17.7 14.3 32 32 32s32-14.3 32-32V413.3l17.6 17.5 0 0c87.5 87.4 229.3 87.4 316.7 0c24.4-24.4 42.1-53.1 52.9-83.7c5.9-16.7-2.9-34.9-19.5-40.8s-34.9 2.9-40.8 19.5c-7.7 21.8-20.2 42.3-37.8 59.8c-62.5 62.5-163.8 62.5-226.3 0L125.7 352H160c17.7 0 32-14.3 32-32s-14.3-32-32-32H48.4c-2.2 0-4.2 .9-5.6 2.3c-1.5 1.4-2.4 3.3-2.5 5.4c0 .5 0 1.1 .1 1.6z"
                                fill="currentColor"
                              />
                            </svg>
                            <span>Flip Board</span>
                          </div>
                          <div className="quick-menu-item" id="copy-pgn">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              viewBox="0 0 384 512"
                            >
                              <path
                                d="M192 0c-41.8 0-77.4 26.7-90.5 64L64 64C28.7 64 0 92.7 0 128L0 448c0 35.3 28.7 64 64 64l256 0c35.3 0 64-28.7 64-64l0-320c0-35.3-28.7-64-64-64l-37.5 0C269.4 26.7 233.8 0 192 0zm0 64a32 32 0 1 1 0 64 32 32 0 1 1 0-64zM72 272a24 24 0 1 1 48 0 24 24 0 1 1 -48 0zm104-16l128 0c8.8 0 16 7.2 16 16s-7.2 16-16 16l-128 0c-8.8 0-16-7.2-16-16s7.2-16 16-16zM72 368a24 24 0 1 1 48 0 24 24 0 1 1 -48 0zm88 0c0-8.8 7.2-16 16-16l128 0c8.8 0 16 7.2 16 16s-7.2 16-16 16l-128 0c-8.8 0-16-7.2-16-16z"
                                fill="currentColor"
                              />
                            </svg>
                            <span>Copy PGN</span>
                          </div>
                        </div>
                      </div>
                      <button id="forward">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 448 512"
                        >
                          <path
                            d="M438.6 278.6c12.5-12.5 12.5-32.8 0-45.3l-160-160c-12.5-12.5-32.8-12.5-45.3 0s-12.5 32.8 0 45.3L338.8 224 32 224c-17.7 0-32 14.3-32 32s14.3 32 32 32l306.7 0L233.4 393.4c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0l160-160z"
                            fill="currentColor"
                          />
                        </svg>
                      </button>
                      <button id="skip-to-end">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 320 512"
                        >
                          <path
                            d="M52.5 440.6c-9.5 7.9-22.8 9.7-34.1 4.4S0 428.4 0 416L0 96C0 83.6 7.2 72.3 18.4 67s24.5-3.6 34.1 4.4l192 160L256 241l0-145c0-17.7 14.3-32 32-32s32 14.3 32 32l0 320c0 17.7-14.3 32-32 32s-32-14.3-32-32l0-145-11.5 9.6-192 160z"
                            fill="currentColor"
                          />
                        </svg>
                      </button>
                    </div>
                  </div>

                  <div className="analysis-overlay active">
                    <div className="analysis-content">
                      <h2>Analyzing Game</h2>
                      <p>Stockfish is analyzing your game...</p>
                      <div className="analysis-progress">
                        <div className="analysis-progress-bar"></div>
                      </div>
                      <div className="fun-fact">
                        Here&apos;s a fun fact: the website is still loading!
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </article>
          </main>
        </div>
      </div>
    </div>
  );
}
