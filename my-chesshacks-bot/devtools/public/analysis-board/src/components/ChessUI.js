"use strict";

import { Chess } from "../../libs/chess.js";

import { Chessboard } from "./board/Chessboard.js";
import { MoveTree } from "./moves/MoveTree.js";
import { MoveNavigator } from "./moves/MoveNavigator.js";
import { EvaluationQueue } from "../evaluation/EvaluationQueue.js";

import { GameStats } from "./report/GameStats.js";
import { GameGraph } from "./report/GameGraph.js";
import { EvaluationBar } from "./board/EvaluationBar.js";
import { EngineLines } from "./moves/EngineLines.js";
import { MoveInformation } from "./moves/MoveInformation.js";
import { Clock } from "./board/Clock.js";
import { DebugInformation } from "./debug/DebugInformation.js";

import { MoveEvaluator } from "../evaluation/MoveEvaluator.js";
import { Classification } from "../classification/MoveClassifier.js";
import { SidebarOverlay } from "./report/SidebarOverlay.js";
import { GameClassifier } from "../classification/GameClassifier.js";
import { SettingsMenu } from "./settings/SettingsMenu.js";

/**
 * Manages UI interactions and board state
 */
export class ChessUI {
  /**
   * Initializes the Chess UI
   * @param {Board} board - The chess board instance
   * @param {MoveTree} moveTree - The move tree instance
   */
  constructor() {
    this._isDestroyed = false;
    this._teardownCallbacks = new Set();
    this._activeLoadToken = null;
    this._hiddenTabs = new Set();
    this._humanMoveListeners = new Set();
    this._humanMoveListenerBound = false;
    this._boardHumanMoveHandler = (move) => this._handleBoardHumanMove(move);

    this.chess = new Chess();
    this.settingsMenu = new SettingsMenu(".settings-menu-container", this);

    // Load the board with settings from cookies
    this.board = new Chessboard(
      "#chessboard",
      {
        theme: {
          boardDarkSquareColor:
            this.settingsMenu.getSettingValue("theme.boardDarkSquareColor") ||
            "rgba(110, 161, 118, 1)",
          boardLightSquareColor:
            this.settingsMenu.getSettingValue("theme.boardLightSquareColor") ||
            "rgba(224, 224, 224, 1)",
          pieceFolderName:
            this.settingsMenu.getSettingValue("pieceTheme") || "cburnett"
        },
        showBoardLabels:
          this.settingsMenu.getSettingValue("theme.boardLabels") === "letter" ||
          true
      },
      this.chess
    );

    this.settingsMenu.init(this.board);

    this.moveTree = new MoveTree();
    this.moveNavigator = new MoveNavigator(this);
    this.evaluationQueue = new EvaluationQueue(this.settingsMenu);

    GameGraph.render();
    GameGraph.setClickCallback((clickedMove) => {
      // Find the move index in the analysis moves array
      const moveIndex = this.analysis?.moves?.indexOf(clickedMove);
      if (moveIndex !== undefined && moveIndex >= 0) {
        // Map to mainline index (add 1 because mainline[0] is root)
        const mainlineIndex = moveIndex + 1;
        if (mainlineIndex < this.moveTree.mainline.length) {
          const targetNode = this.moveTree.mainline[mainlineIndex];
          this.moveNavigator.handleTreeNodeClick(targetNode);
        }
      }
    });

    GameStats.render();

    // Initialize evaluation bar and panels based on showClassification setting
    const showClassification =
      this.settingsMenu.getSettingValue("showClassification");
    const $evalBar = $(".eval-bar");
    if (showClassification) {
      $evalBar.show();
      EvaluationBar.updateEvaluationBar();
    } else {
      $evalBar.hide();
    }

    MoveInformation.updateMoveInfo(
      this.moveTree.mainline[0],
      this.moveTree.mainline[0],
      showClassification
    );
    EngineLines.updateEngineLines(
      this.moveTree.mainline[0],
      this.moveTree,
      (node) => this.moveNavigator.handleTreeNodeClick(node),
      (node, resultFen, prevFen) =>
        this.moveNavigator.queueMoveForEvaluation(node, resultFen, prevFen),
      showClassification
    );

    // Initialize clocks to default state
    Clock.resetClocks();

    // Initialize debug information
    DebugInformation.updateDebugInfo(
      this.moveTree.mainline[0],
      this.moveTree.mainline[0]
    );
  }

  addTeardownCallback(callback) {
    if (typeof callback === "function") {
      this._teardownCallbacks.add(callback);
    }
  }

  onHumanMove(callback) {
    if (typeof callback !== "function") {
      console.warn("onHumanMove expects a function callback.");
      return () => {};
    }

    if (this._isDestroyed) {
      console.warn(
        "Cannot register human move callback after ChessUI is destroyed."
      );
      return () => {};
    }

    this._humanMoveListeners.add(callback);

    if (this.eventHandlersSetup) {
      this._bindHumanMoveListenerIfNeeded();
    }

    return () => {
      this._humanMoveListeners.delete(callback);

      if (this._humanMoveListeners.size === 0) {
        if (this._humanMoveListenerBound && this.board?.off) {
          this.board.off("usermove", this._boardHumanMoveHandler);
        }
        this._humanMoveListenerBound = false;
      }
    };
  }

  getGamePGN() {
    // go through the move tree (up to the current node) and build the PGN string
    if (!this.moveTree || !this.moveTree.currentNode) {
      return "";
    }

    const currentNode = this.moveTree.currentNode;

    // Get the moves from root to current node
    const moves = this.moveTree.getMovesToNode(currentNode.id);

    if (moves.length === 0) {
      return "";
    }

    // Build PGN string from the moves
    const pgnMoves = [];
    let whiteMove = null;
    let moveNumber = 1;

    for (const moveData of moves) {
      if (!moveData.san || !moveData.move) continue;

      if (moveData.move.color === "w") {
        // White move - start a new move number
        // If we have a pending white move, output it first
        if (whiteMove) {
          pgnMoves.push(`${moveNumber}. ${whiteMove}`);
          moveNumber++;
        }
        whiteMove = moveData.san;
      } else if (moveData.move.color === "b") {
        // Black move - complete the turn
        if (whiteMove) {
          pgnMoves.push(`${moveNumber}. ${whiteMove} ${moveData.san}`);
          whiteMove = null;
          moveNumber++;
        } else {
          // Black move without preceding white move (e.g., starting from a position where black moves first)
          pgnMoves.push(`${moveNumber}... ${moveData.san}`);
          moveNumber++;
        }
      }
    }

    // Handle case where we end on a white move
    if (whiteMove) {
      pgnMoves.push(`${moveNumber}. ${whiteMove}`);
    }

    return pgnMoves.join(" ");
  }

  hideTab(tabName) {
    if (this._isDestroyed) return false;

    const resolved = this._resolveTabElements(tabName);
    if (!resolved) {
      console.warn(`hideTab: Unable to resolve tab "${tabName}".`);
      return false;
    }

    const { tabName: actualName, $button, $panel } = resolved;

    if (this._hiddenTabs.has(actualName)) {
      return true;
    }

    const wasActive =
      ($button && $button.length && $button.hasClass("active")) ||
      ($panel && $panel.length && $panel.hasClass("active"));

    this._setTabVisibility(actualName, false, resolved);
    this._hiddenTabs.add(actualName);

    if (wasActive) {
      this._activateFirstVisibleTab();
    }

    return true;
  }

  playUciMove(uciMove) {
    if (this._isDestroyed) return null;
    if (!this.board || !this.moveNavigator) return null;
    if (typeof uciMove !== "string") {
      console.warn("playUciMove expects a UCI string.");
      return null;
    }

    const trimmed = uciMove.trim().toLowerCase();
    if (trimmed.length < 4) {
      console.warn(`Invalid UCI move: ${uciMove}`);
      return null;
    }

    const from = trimmed.substring(0, 2);
    const to = trimmed.substring(2, 4);
    const promotion = trimmed.length > 4 ? trimmed.substring(4) : null;
    const promotionPiece = promotion ? promotion[0] : null;
    const squarePattern = /^[a-h][1-8]$/;

    if (!squarePattern.test(from) || !squarePattern.test(to)) {
      console.warn(`Invalid squares in UCI move: ${uciMove}`);
      return null;
    }

    const fenBefore = this.getCurrentFen() || this.board?.chess?.fen?.();
    const moveInput = { from, to };
    if (promotionPiece) {
      moveInput.promotion = promotionPiece;
    }

    const moveResult = this.board.move(
      moveInput,
      true,
      null,
      fenBefore,
      false,
      promotionPiece || null,
      false
    );

    if (!moveResult) {
      console.warn(`Failed to play UCI move: ${uciMove}`);
      if (fenBefore && typeof this.board.fen === "function") {
        this.board.fen(fenBefore);
      }
      return null;
    }

    if (!moveResult.before && fenBefore) {
      moveResult.before = fenBefore;
    }
    if (!moveResult.after && this.board?.chess?.fen) {
      moveResult.after = this.board.chess.fen();
    }
    if (!moveResult.promotion && promotionPiece) {
      moveResult.promotion = promotionPiece;
    }

    this.moveNavigator.handleUserMove(moveResult);
    return moveResult;
  }

  getCurrentFen() {
    if (!this.moveTree || !this.moveTree.currentNode) {
      return this.board?.chess?.fen?.() || this.chess?.fen?.() || null;
    }

    const currentNode = this.moveTree.currentNode;
    if (currentNode.move?.after) {
      return currentNode.move.after;
    }

    if (currentNode.fen) {
      return currentNode.fen;
    }

    return this.board?.chess?.fen?.() || this.chess?.fen?.() || null;
  }

  _handleBoardHumanMove(move) {
    if (this._isDestroyed || this._humanMoveListeners.size === 0) return;

    const currentNode = this.moveTree?.currentNode || null;
    const previousNode = this.moveTree?.getPreviousMove
      ? this.moveTree.getPreviousMove()
      : null;

    const beforeFen =
      move?.before ||
      previousNode?.move?.after ||
      previousNode?.fen ||
      this.moveTree?.mainline?.[0]?.fen ||
      null;

    const afterFen =
      move?.after ||
      currentNode?.move?.after ||
      currentNode?.fen ||
      this.board?.chess?.fen?.() ||
      null;

    const moveForCallback = this.getGamePGN();

    this._humanMoveListeners.forEach((listener) => {
      try {
        listener(moveForCallback);
      } catch (error) {
        console.error("Error in human move listener:", error);
      }
    });
  }

  _resolveTabElements(tabName) {
    if (typeof tabName !== "string") return null;

    const normalized = tabName.trim();
    if (!normalized) return null;

    const lower = normalized.toLowerCase();

    let $button = $(".tab-button").filter((_, el) => {
      const value = $(el).data("tab");
      return typeof value === "string" && value.toLowerCase() === lower;
    });

    if ($button.length) {
      const $buttonEl = $button.first();
      const actual = $buttonEl.data("tab");
      const $panelEl = $(`#${actual}-tab`);
      return { tabName: actual, $button: $buttonEl, $panel: $panelEl };
    }

    const $panel = $(".tab-panel").filter((_, el) => {
      const id = (el.id || "").toLowerCase();
      return id === `${lower}-tab`;
    });

    if ($panel.length) {
      const panelId = $panel[0].id;
      const actual = panelId.replace(/-tab$/i, "");
      const $buttonEl = $(".tab-button")
        .filter((_, el) => {
          const value = $(el).data("tab");
          return (
            typeof value === "string" &&
            value.toLowerCase() === actual.toLowerCase()
          );
        })
        .first();
      return { tabName: actual, $button: $buttonEl, $panel: $panel.first() };
    }

    return null;
  }

  _setTabVisibility(tabName, isVisible, resolvedElements = null) {
    const resolved = resolvedElements || this._resolveTabElements(tabName);
    if (!resolved) return false;

    const { $button, $panel } = resolved;

    if ($button && $button.length) {
      if (isVisible) {
        $button.removeAttr("hidden");
        $button.attr("aria-hidden", "false");
        $button.prop("disabled", false);
      } else {
        $button.attr("hidden", "hidden");
        $button.attr("aria-hidden", "true");
        $button.prop("disabled", true);
        $button.removeClass("active");
      }
    }

    if ($panel && $panel.length) {
      if (isVisible) {
        $panel.removeAttr("hidden");
        $panel.attr("aria-hidden", "false");
      } else {
        $panel.attr("hidden", "hidden");
        $panel.attr("aria-hidden", "true");
        $panel.removeClass("active");
      }
    }

    return true;
  }

  _activateFirstVisibleTab() {
    const $availableButtons = $(".tab-button").filter((_, el) => {
      const $el = $(el);
      return $el.is(":visible") && !$el.prop("disabled");
    });

    if (!$availableButtons.length) {
      $(".tab-button").removeClass("active");
      $(".tab-panel").removeClass("active");
      return;
    }

    const $targetButton = $availableButtons.first();
    const tabName = $targetButton.data("tab");

    $targetButton.trigger("click");

    if (!$targetButton.hasClass("active")) {
      $(".tab-button").removeClass("active");
      $(".tab-panel").removeClass("active");

      $targetButton.addClass("active");
      const $targetPanel = $(`#${tabName}-tab`);
      if ($targetPanel.length) {
        $targetPanel.removeAttr("hidden");
        $targetPanel.attr("aria-hidden", "false");
        $targetPanel.addClass("active");
      }

      if (tabName === "report" && GameGraph?.render) {
        try {
          GameGraph.render();
        } catch (error) {
          console.error(
            "Error re-rendering GameGraph while activating tab:",
            error
          );
        }
      }
    }
  }

  _bindHumanMoveListenerIfNeeded() {
    if (
      this._isDestroyed ||
      this._humanMoveListenerBound ||
      !this.board ||
      this._humanMoveListeners.size === 0
    ) {
      return;
    }

    if (typeof this.board.on === "function") {
      this.board.on("usermove", this._boardHumanMoveHandler);
      this._humanMoveListenerBound = true;
    }
  }

  /**
   * Refreshes the classification display when the showClassification setting is toggled
   */
  refreshClassificationDisplay() {
    if (this._isDestroyed) return;

    const showClassification =
      this.settingsMenu.getSettingValue("showClassification");
    const currentNode = this.moveTree.currentNode;

    // Re-render the move tree with the new setting
    this.moveTree.render(
      "move-tree",
      (node) => {
        this.moveNavigator.handleTreeNodeClick(node);
      },
      showClassification
    );

    // Re-highlight the current move after re-rendering
    this.moveTree.updateCurrentMove(currentNode.id);

    // Update the current move's classification on the board
    this.moveTree.updateNodeClassification(
      currentNode,
      this.board,
      showClassification
    );

    // Update the move info and engine lines panels
    const prevNode = this.moveTree.getPreviousMove();
    MoveInformation.updateMoveInfo(currentNode, prevNode, showClassification);

    EngineLines.updateEngineLines(
      currentNode,
      this.moveTree,
      (node) => this.moveNavigator.handleTreeNodeClick(node),
      (node, resultFen, prevFen) =>
        this.moveNavigator.queueMoveForEvaluation(node, resultFen, prevFen),
      showClassification
    );

    // Show/hide the evaluation bar
    const $evalBar = $(".eval-bar");
    if (showClassification) {
      $evalBar.show();
    } else {
      $evalBar.hide();
    }
  }

  /**
   * Refreshes the probability arrows when the showProbabilityArrows setting is toggled
   */
  refreshProbabilityArrows() {
    if (this._isDestroyed) return;

    const showProbabilityArrows = this.settingsMenu.getSettingValue(
      "showProbabilityArrows"
    );
    const showClassification =
      this.settingsMenu.getSettingValue("showClassification");
    const currentNode = this.moveTree.currentNode;
    const prevNode = this.moveTree.getPreviousMove();

    // Update probability arrows based on the setting
    if (showProbabilityArrows) {
      DebugInformation.updateProbabilityArrows(
        this.board,
        currentNode,
        prevNode
      );
    } else {
      this.board.clearProbabilityArrows();
    }

    // Re-apply classifications if they're enabled (since clearing arrows might affect the board)
    if (showClassification && currentNode) {
      this.moveTree.updateNodeClassification(
        currentNode,
        this.board,
        showClassification
      );
    }
  }

  /**
   * Refreshes the best move arrow when the showBestMoveArrow setting is toggled
   */
  refreshBestMoveArrow() {
    if (this._isDestroyed) return;

    const showBestMoveArrow =
      this.settingsMenu.getSettingValue("showBestMoveArrow");
    const showClassification =
      this.settingsMenu.getSettingValue("showClassification");
    const showProbabilityArrows = this.settingsMenu.getSettingValue(
      "showProbabilityArrows"
    );
    const currentNode = this.moveTree.currentNode;
    const prevNode = this.moveTree.getPreviousMove();

    // Clear all arrows first
    this.board.clearBoardElements();

    // Re-add best move arrow if enabled
    if (showClassification && showBestMoveArrow) {
      if (prevNode?.evaluatedMove?.lines?.length > 0) {
        const bestLine = prevNode.evaluatedMove.lines.find(
          (line) => line.id === 1
        );
        if (bestLine) {
          const bestMoveUci =
            bestLine.uciMove ||
            (bestLine.pv && bestLine.pv.length > 0 ? bestLine.pv[0] : null);
          if (bestMoveUci) {
            const fromIdx = this.board.algebraicToIndex(
              bestMoveUci.substring(0, 2),
              this.board.flipped
            );
            const toIdx = this.board.algebraicToIndex(
              bestMoveUci.substring(2, 4),
              this.board.flipped
            );
            this.board.createArrow(
              this.board.getSquare(fromIdx),
              this.board.getSquare(toIdx)
            );
          }
        }
      }
    }

    // Re-add probability arrows if enabled
    if (showProbabilityArrows) {
      DebugInformation.updateProbabilityArrows(
        this.board,
        currentNode,
        prevNode
      );
    }

    // Re-apply classifications if they're enabled
    if (showClassification && currentNode) {
      this.moveTree.updateNodeClassification(
        currentNode,
        this.board,
        showClassification
      );
    }
  }

  async load(game) {
    if (this._isDestroyed) return;

    const loadToken = Symbol("load");
    this._activeLoadToken = loadToken;

    this.moveNavigator.handleRestart();

    this.game = game;

    // Load PGN into the main Chess instance first
    this.chess.loadPgn(this.game.pgn);
    this.chess.reset(); // Reset to starting position for analysis

    this.moveTree.buildFromPGN(this.game.pgn, this.chess);

    // Set initial clocks before analysis starts
    Clock.setInitialClocks(this.moveTree, this.game.pgn);

    SidebarOverlay.show();
    SidebarOverlay.startFactCycling();
    SidebarOverlay.updateEvaluationProgress(0);

    this.board.fen(this.moveTree.mainline[0].fen);

    // Flip to face the username player
    const userIsBlack =
      this.game.username.toLowerCase() === this.game.black.name.toLowerCase();
    if (userIsBlack) {
      this.moveNavigator.handleFlipBoard();
    }

    $(".analysis-overlay").addClass("active");
    $(".tab-content, .bottom-content").addClass("blur-content");
    this.board.setOption({ isInteractive: false });

    const engineType = this.settingsMenu.getSettingValue("engineType");
    const engineDepth = this.settingsMenu.getSettingValue("engineDepth") || 14;

    const analysis = await MoveEvaluator.analyzeGame(
      this.game,
      (progress) => {
        if (!this._isDestroyed) {
          SidebarOverlay.updateEvaluationProgress(progress);
        }
      },
      { engineType, engineDepth }
    );

    if (this._isDestroyed || this._activeLoadToken !== loadToken) {
      return;
    }

    this.board.setOption({ isInteractive: true });

    // Store analysis for click callback
    this.analysis = analysis;

    const graphedMoves = analysis.moves.map((move) => move.graph / 100);

    const classify = new GameClassifier();
    const gameClass = classify.classifyGame(
      graphedMoves,
      userIsBlack ? "w" : "b",
      this.game.result
    );
    $(".game-info").empty().append(`<p>${gameClass.message}</p>`);

    SidebarOverlay.hide();
    SidebarOverlay.stopFactCycling();

    MoveEvaluator.applyClassificationsToMoveTree(
      this.moveTree,
      analysis.moves,
      this.game.pgn
    );
    GameGraph.setAnalysis(analysis);
    GameStats.render(".game-stats", analysis, game.white.name, game.black.name);

    const showClassification =
      this.settingsMenu.getSettingValue("showClassification");
    this.moveTree.render(
      "move-tree",
      (node) => {
        this.moveNavigator.handleTreeNodeClick(node);
      },
      showClassification
    );

    $(".analysis-overlay").removeClass("active");
    $(".tab-content, .bottom-content").removeClass("blur-content");

    // Initialize clocks
    Clock.updateFromMoveTree(this.moveTree, this.board.flipped, this.game?.pgn);

    if (!this.eventHandlersSetup) {
      this.moveNavigator.setupEventHandlers();
      this.eventHandlersSetup = true;
    }

    this._bindHumanMoveListenerIfNeeded();

    if (this._activeLoadToken === loadToken) {
      this._activeLoadToken = null;
    }
  }

  destroy() {
    if (this._isDestroyed) return;

    this._isDestroyed = true;
    this._activeLoadToken = null;

    const teardownCallbacks = Array.from(this._teardownCallbacks);
    this._teardownCallbacks.clear();
    teardownCallbacks.forEach((callback) => {
      try {
        callback();
      } catch (error) {
        console.error("Error running teardown callback:", error);
      }
    });

    if (this.moveNavigator?.destroy) {
      this.moveNavigator.destroy();
    }
    this.moveNavigator = null;

    if (this.evaluationQueue?.destroy) {
      this.evaluationQueue.destroy();
    }
    this.evaluationQueue = null;

    if (this._humanMoveListenerBound && this.board?.off) {
      this.board.off("usermove", this._boardHumanMoveHandler);
    }
    this._humanMoveListenerBound = false;

    if (this.board?.destroy) {
      this.board.destroy();
    }
    this.board = null;

    if (this.settingsMenu?.destroy) {
      this.settingsMenu.destroy();
    }
    this.settingsMenu = null;

    if (this.moveTree?.destroy) {
      this.moveTree.destroy();
    }
    this.moveTree = null;

    if (typeof GameGraph.destroy === "function") {
      GameGraph.destroy();
    } else {
      GameGraph.setClickCallback(null);
    }

    if (typeof SidebarOverlay.reset === "function") {
      SidebarOverlay.reset();
    } else {
      SidebarOverlay.stopFactCycling();
    }

    $(".analysis-overlay, .board-overlay").removeClass("active");
    $(".tab-content, .bottom-content").removeClass("blur-content");
    $(".analysis-progress-bar").css("width", "0%");
    $(".progress-percentage").text("");
    $(".game-info").empty();
    $(".game-stats").empty();
    $(".move-info").empty();
    $(".engine-lines").empty();
    $("#move-tree").empty();
    $(".debug-probabilities").empty();
    $(".debug-logs-section").empty();
    $(".quick-notification").remove();

    EvaluationBar.updateEvaluationBar();
    Clock.resetClocks();

    if (this._hiddenTabs.size > 0) {
      this._hiddenTabs.forEach((tabName) => {
        this._setTabVisibility(tabName, true);
      });
      this._hiddenTabs.clear();
    }

    this._humanMoveListeners.clear();

    this.chess = null;
    this.game = null;
    this.analysis = null;
  }

  async cacheClassifications() {
    const classifications = Object.entries(Classification);
    const loadPromises = [];

    // Set up all classification loading in parallel
    $.each(classifications, (_, classif) => {
      const path = classif[1].src;
      const type = classif[1].type;

      const promise = $.ajax({
        url: path,
        dataType: "text"
      })
        .then((svgText) => {
          // Convert SVG to base64 data URI
          const base64 = btoa(unescape(encodeURIComponent(svgText)));
          const dataUri = `data:image/svg+xml;base64,${base64}`;

          // Create and cache the image
          const $img = $("<img>", {
            src: dataUri,
            class: "move-icon",
            alt: type
          })[0]; // Get the DOM element

          Classification[classif[0]].cachedImg = $img;
          return type;
        })
        .catch((error) => {
          console.error(`Failed to load ${path}:`, error);
          return null;
        });

      loadPromises.push(promise);
    });

    // Wait for all classifications to load
    await Promise.allSettled(loadPromises);
  }
}
