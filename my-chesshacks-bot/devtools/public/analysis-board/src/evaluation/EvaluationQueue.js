import { Engine } from "./Engine.js";
import { MoveEvaluator } from "./MoveEvaluator.js";
import { MoveClassifier } from "../classification/MoveClassifier.js";

/**
 * Manages a queue of moves to be evaluated and handles background processing
 */
export class EvaluationQueue {
  constructor(settingsMenu) {
    this.queue = [];
    this.currentEvaluation = null;
    this.isProcessing = false;
    this.processedMoves = new Map(); // Maps node IDs to evaluation results
    this.displayProgressBar = true;
    this.settingsMenu = settingsMenu;
    this.activeEngines = new Set();
    this._isDestroyed = false;
  }

  /**
   * Adds a move to the evaluation queue
   * @param {Object} node - The move tree node to evaluate
   * @param {string} fen - The FEN string after the move
   * @param {string} previousFen - The FEN string before the move
   * @param {Function} callback - Function to call when evaluation is complete
   * @param {MoveTree} moveTree - Optional reference to the move tree for priority calculation
   */
  addToQueue(node, fen, previousFen, callback, moveTree) {
    if (this._isDestroyed) return;

    const nodeId = node.id;

    // Skip if already queued or processed
    if (this.isNodeQueued(nodeId) || this.processedMoves.has(nodeId)) return;

    // Add to queue with priority (lower number = higher priority)
    const priority = moveTree && nodeId === moveTree.currentNode.id ? 0 : 1;
    const queueItem = {
      node,
      fen,
      previousFen,
      callback,
      priority,
      timeAdded: Date.now(),
      moveTree
    };

    // Insert in priority order
    const insertIndex = this.queue.findIndex(
      (item) => item.priority > priority
    );
    insertIndex === -1
      ? this.queue.push(queueItem)
      : this.queue.splice(insertIndex, 0, queueItem);

    // Start processing if not already running
    if (!this.isProcessing) this.processQueue();
  }

  /**
   * Processes the evaluation queue
   */
  async processQueue() {
    if (this._isDestroyed || this.queue.length === 0 || this.isProcessing)
      return;

    this.updateMiniEvaluationProgress(0);

    this.isProcessing = true;
    const item = this.queue.shift();
    if (!item || this._isDestroyed) {
      this.isProcessing = false;
      return;
    }
    this.currentEvaluation = item;

    let engine = null;

    try {
      // Check if previous position has already been evaluated
      let prevLines = this.findPreviousLines(item);

      // If no previous lines found, evaluate the previous position
      if (!prevLines) {
        prevLines =
          (await MoveEvaluator.tryCloudEvaluation(item.previousFen)) ||
          (await this.evaluateWithEngine(item.previousFen, 12, 0, 100));
      }

      // Evaluate current position
      let lines = await MoveEvaluator.tryCloudEvaluation(item.fen);

      if (!lines || lines.length < 2) {
        const engineType =
          this.settingsMenu?.getSettingValue("engineType") ||
          "stockfish-17-lite";
        engine = new Engine({ engineType: engineType });
        this.activeEngines.add(engine);
        const depth =
          this.settingsMenu?.getSettingValue("variationEngineDepth") || 16;
        lines = await this.evaluateWithEngine(item.fen, depth, 0, 100, engine);
      }

      // Create and store result
      const result = {
        move: {
          fen: item.fen,
          lines: lines,
          uciMove: item.node.move
            ? `${item.node.move.from}${item.node.move.to}`
            : "",
          engine: engine ? engine.engine.name : "Cloud"
        },
        previous: { fen: item.previousFen, lines: prevLines }
      };

      this.processedMoves.set(item.node.id, result);

      // Process callback if provided
      if (item.callback && !this._isDestroyed) {
        const movesUpToCurrent = this.getMovesUpToCurrent(
          item.node,
          item.moveTree
        );
        const classification = MoveClassifier.classifyMove(
          result.move,
          result.previous,
          movesUpToCurrent
        );

        item.callback({
          classification,
          uciMove: item.node.move,
          fen: item.fen,
          lines,
          engine: engine ? engine.engine.name : "Cloud"
        });
      }

      if (engine) {
        try {
          engine.abort?.();
          engine.terminate?.();
        } finally {
          this.activeEngines.delete(engine);
        }
      }

      this.updateMiniEvaluationProgress(100);
    } catch (error) {
      console.error("Error during evaluation:", error);
      if (engine) {
        try {
          engine.abort?.();
          engine.terminate?.();
        } finally {
          this.activeEngines.delete(engine);
        }
      }
    }

    // Clear current evaluation and continue with queue
    this.currentEvaluation = null;
    this.isProcessing = false;

    if (!this._isDestroyed && this.queue.length > 0) this.processQueue();
  }

  /**
   * Evaluates a position using the engine with progress tracking
   * @private
   */
  async evaluateWithEngine(
    fen,
    depth,
    startProgress,
    endProgress,
    engine = null
  ) {
    if (this._isDestroyed) return [];

    let localEngineCreated = false;

    if (!engine) {
      // Get engine type from settings
      const engineType =
        this.settingsMenu?.getSettingValue("engineType") || "stockfish-17-lite";
      engine = new Engine({ engineType: engineType });
      localEngineCreated = true;
      this.activeEngines.add(engine);
    }

    try {
      return await engine.evaluate(fen, depth, false, (progress) => {
        if (this._isDestroyed) {
          engine.abort?.();
          return;
        }
        const scaledProgress =
          startProgress +
          (progress.percent * (endProgress - startProgress)) / 100;
        this.updateMiniEvaluationProgress(Math.round(scaledProgress));
      });
    } finally {
      if (localEngineCreated || this._isDestroyed) {
        try {
          engine.abort?.();
        } catch (error) {
          console.warn("Error aborting engine:", error);
        }
        try {
          engine.terminate?.();
        } catch (error) {
          console.warn("Error terminating engine:", error);
        }
        this.activeEngines.delete(engine);
      }
    }
  }

  /**
   * Finds previous evaluation lines for a node
   * @private
   */
  findPreviousLines(item) {
    if (!item.moveTree || !item.node) return null;

    let parentNode = null;
    let parentNodeId = null;

    // Get parent node ID
    if (item.node.isMainline && typeof item.node.parentIndex === "number") {
      parentNode = item.moveTree.mainline[item.node.parentIndex];
      parentNodeId = parentNode?.id;
    } else if (item.node.parentId) {
      parentNodeId = item.node.parentId;
      parentNode = item.moveTree.nodeMap.get(parentNodeId);
    }

    if (!parentNodeId) return null;

    // Try to get stored evaluation for parent node
    const parentResult = this.processedMoves.get(parentNodeId);
    if (parentResult?.move?.lines) return parentResult.move.lines;

    // Try evaluation from the move tree
    return parentNode?.evaluatedMove?.lines || null;
  }

  /**
   * Gets the list of moves up to the current node
   * @private
   */
  getMovesUpToCurrent(node, moveTree) {
    if (!moveTree) return [];

    const moves = [];
    let currentNode = node;

    while (currentNode) {
      if (currentNode.move) moves.unshift(currentNode.san);

      if (
        currentNode.isMainline &&
        typeof currentNode.parentIndex === "number"
      ) {
        currentNode = moveTree.mainline[currentNode.parentIndex];
      } else if (currentNode.parentId) {
        currentNode = moveTree.nodeMap.get(currentNode.parentId);
      } else {
        currentNode = null;
      }
    }

    return moves;
  }

  /**
   * Checks if a node is in the evaluation queue or currently being evaluated
   * @param {string} nodeId - The ID of the node to check
   * @returns {boolean} True if the node is queued or being evaluated
   */
  isNodeQueued(nodeId) {
    return (
      this.queue.some((item) => item.node.id === nodeId) ||
      this.currentEvaluation?.node.id === nodeId
    );
  }

  /**
   * Gets the evaluation result for a specific node
   * @param {string} nodeId - The ID of the node
   * @returns {Object|null} The evaluation result or null if not evaluated
   */
  getResult(nodeId) {
    return this.processedMoves.get(nodeId) || null;
  }

  /**
   * Updates the evaluation progress bar
   * @param {number|object} progress - Progress percentage (0-100) or progress object with depth info
   */
  updateMiniEvaluationProgress(progress) {
    if (this._isDestroyed) return;

    if (!this.displayProgressBar) return;

    // Convert progress object to percentage if needed
    let percentage =
      typeof progress === "object" && progress !== null
        ? progress.percent || 0
        : progress;

    // Simple progress bar for individual move evaluation
    const progressBar = $(".evaluation-progress-bar");
    progressBar.addClass("visible");
    progressBar.css("opacity", "1");

    progressBar.css("width", percentage + "%");

    // Hide when complete
    if (percentage >= 100) {
      progressBar.css("opacity", "0");
      progressBar.css("width", "0%");
    }
  }

  destroy() {
    if (this._isDestroyed) return;

    this._isDestroyed = true;
    this.queue = [];
    this.currentEvaluation = null;
    this.isProcessing = false;
    this.displayProgressBar = false;
    this.processedMoves.clear();

    this.activeEngines.forEach((engine) => {
      try {
        engine.abort?.();
      } catch (error) {
        console.warn("Error aborting engine during teardown:", error);
      }
      try {
        engine.terminate?.();
      } catch (error) {
        console.warn("Error terminating engine during teardown:", error);
      }
    });

    this.activeEngines.clear();
  }
}
