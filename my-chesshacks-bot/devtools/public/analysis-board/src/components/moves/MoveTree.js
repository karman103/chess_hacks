import { Chess } from "./../../../libs/chess.js";
import { Classification } from "../../classification/MoveClassifier.js";

export class MoveTree {
  constructor() {
    this.mainline = [
      {
        id: "root",
        moveNumber: 1,
        san: null,
        fen: null,
        children: [],
        isMainline: true,
      },
    ];
    this.nodeMap = new Map().set("root", this.mainline[0]);
    this.currentNode = this.mainline[0];
    this.final = null;
    this.idCounter = 0;
    this.currentIndex = 0;
    this.clockData = new Map(); // Store clock data for each position
  }

  /**
   * Extracts base64 encoded comments from PGN string
   * @param {string} pgn - The PGN string
   * @returns {Array} Array of decoded comment data objects
   */
  extractCommentData(pgn) {
    const commentData = [];

    // Remove header from PGN to get only the moves
    const headerRegex = /^\[[\s\S]*?\]\s*$/gm;
    const moveText = pgn.replace(headerRegex, "").trim();

    // Find all comments in curly braces
    const commentRegex = /\{([^}]+)\}/g;
    const commentMatches = [...moveText.matchAll(commentRegex)];

    if (commentMatches.length === 0) {
      return commentData;
    }

    // Extract moves with their positions
    const moveRegex = /\d+\.\s*([^\s{]+)(?:\s+\{[^}]*\})?\s*([^\s{]+)?(?:\s+\{[^}]*\})?/g;
    let moveIndex = 0;
    let commentIndex = 0;

    while (true) {
      const match = moveRegex.exec(moveText);
      if (!match) break;

      const whiteMove = match[1];
      const blackMove = match[2];

      if (whiteMove) {
        const whiteMovePos = match.index + match[0].indexOf(whiteMove);
        
        // Find the next comment after this move
        while (commentIndex < commentMatches.length) {
          if (commentMatches[commentIndex].index > whiteMovePos) {
            try {
              const commentContent = commentMatches[commentIndex][1].trim();
              // Try to decode as base64
              const decoded = atob(commentContent);
              const parsed = JSON.parse(decoded);
              
              commentData[moveIndex] = {
                moveIndex: moveIndex,
                whiteTimeLeft: parsed.whiteTimeLeft,
                blackTimeLeft: parsed.blackTimeLeft,
                probabilities: parsed.probabilities,
                logs: parsed.logs
              };
            } catch (e) {
              // Not a base64 encoded comment or not valid JSON, skip it
            }
            commentIndex++;
            break;
          }
          commentIndex++;
        }
        moveIndex++;
      }

      if (blackMove) {
        const blackMovePos = match.index + match[0].indexOf(blackMove);
        
        // Find the next comment after this move
        while (commentIndex < commentMatches.length) {
          if (commentMatches[commentIndex].index > blackMovePos) {
            try {
              const commentContent = commentMatches[commentIndex][1].trim();
              // Try to decode as base64
              const decoded = atob(commentContent);
              const parsed = JSON.parse(decoded);
              
              commentData[moveIndex] = {
                moveIndex: moveIndex,
                whiteTimeLeft: parsed.whiteTimeLeft,
                blackTimeLeft: parsed.blackTimeLeft,
                probabilities: parsed.probabilities,
                logs: parsed.logs
              };
            } catch (e) {
              // Not a base64 encoded comment or not valid JSON, skip it
            }
            commentIndex++;
            break;
          }
          commentIndex++;
        }
        moveIndex++;
      }
    }

    return commentData;
  }

  buildFromPGN(pgn, chess) {
    // Use the provided chess instance or create a new one
    const chessInstance = chess || new Chess();

    // Extract comment data (includes clock times and debug info)
    const commentData = this.extractCommentData(pgn);

    chessInstance.loadPgn(pgn);
    const history = chessInstance.history({ verbose: true });
    chessInstance.reset();

    this.mainline = [
      {
        id: "root",
        moveNumber: null,
        san: null,
        fen: chessInstance.fen(),
        children: [],
        isMainline: true,
      },
    ];
    this.nodeMap = new Map().set("root", this.mainline[0]);
    this.currentNode = this.mainline[0];
    this.currentIndex = 0;

    let moveNumber = 1,
      isWhiteTurn = true;

    for (const [index, move] of history.entries()) {
      const nodeId = `move_${moveNumber}_${isWhiteTurn ? "w" : "b"}_${move.san.replace(/[+#]/g, (m) => (m === "+" ? "check" : "mate"))}`;

      const node = {
        id: nodeId,
        moveNumber: isWhiteTurn ? moveNumber : moveNumber + 0.5,
        san: move.san,
        move: move,
        fen: null,
        children: [],
        isMainline: true,
        parentIndex: this.mainline.length - 1,
      };

      // Add comment data if available (clock times, probabilities, logs)
      const comment = commentData.find((c) => c.moveIndex === index);
      if (comment) {
        node.whiteTimeLeft = comment.whiteTimeLeft;
        node.blackTimeLeft = comment.blackTimeLeft;
        node.probabilities = comment.probabilities;
        node.logs = comment.logs;
      }

      chessInstance.move(move.lan);
      node.fen = chessInstance.fen();

      this.mainline.push(node);
      this.nodeMap.set(node.id, node);

      if (!isWhiteTurn) moveNumber++;
      isWhiteTurn = !isWhiteTurn;
      this.final = node;
    }

    this.currentNode = this.mainline[0];

    return history;
  }

  updateClassification(nodeId, move) {
    const node = this.nodeMap.get(nodeId);
    if (!node) return;

    node.classification = move.classification.type;
    node.evaluatedMove = move;

    if (move.lines?.length > 0) {
      const topLine = move.lines.find((line) => line.id === 1);
      if (topLine) {
        node.evalScore = topLine.score;
        node.evalType = topLine.type || "cp";
      }
    }
  }

  findExistingMove(parentNode, move) {
    if (parentNode.isMainline) {
      const parentIndex = this.getNodeIndex(parentNode);
      if (parentIndex !== -1 && parentIndex + 1 < this.mainline.length) {
        const nextMove = this.mainline[parentIndex + 1];
        if (
          nextMove.move &&
          nextMove.move.from === move.from &&
          nextMove.move.to === move.to &&
          nextMove.move.promotion === move.promotion
        ) {
          return nextMove;
        }
      }
    }

    return parentNode.children.find(
      (child) =>
        child.move?.from === move.from &&
        child.move?.to === move.to &&
        child.move?.promotion === move.promotion,
    );
  }

  getNodeIndex(node) {
    return this.mainline.findIndex((n) => n.id === node.id);
  }

  addMove(move, parentId) {
    const chess = new Chess();
    const parent = this.nodeMap.get(parentId);
    if (!parent) return null;

    const existingNode = this.findExistingMove(parent, move);
    if (existingNode) return existingNode;

    const originalFen = chess.fen();
    chess.load(parent.fen);

    const isWhiteTurn = chess.turn() === "w";
    const moveNumber = chess.moveNumber();
    const parentIndex = this.getNodeIndex(parent);
    const isMainline =
      parent.isMainline && parentIndex === this.mainline.length - 1;

    this.idCounter++;
    let nodeId = `${isMainline ? "move" : "var"}_${moveNumber}_${isWhiteTurn ? "w" : "b"}_${move.san.replace(/[+#]/g, (m) => (m === "+" ? "check" : "mate"))}${isMainline ? "" : "_" + this.idCounter}`;

    if (isMainline && this.nodeMap.has(nodeId)) {
      nodeId = `${nodeId}_${this.idCounter}`;
    }

    const node = {
      id: nodeId,
      moveNumber: isWhiteTurn ? moveNumber : moveNumber + 0.5,
      san: move.san,
      move: move,
      fen: null,
      children: [],
      isMainline: isMainline,
      parentId: parentId,
    };

    chess.move(move);
    node.fen = chess.fen();

    if (isMainline) {
      node.parentIndex = parentIndex;
      this.mainline.push(node);
      this.final = node;
    } else {
      parent.children.push(node);
    }

    this.nodeMap.set(node.id, node);
    chess.load(originalFen);

    return node;
  }

  navigateTo(nodeId) {
    const node = this.nodeMap.get(nodeId);
    if (node) {
      this.currentNode = node;
      const nodeIndex = this.getNodeIndex(node);
      if (nodeIndex !== -1) this.currentIndex = nodeIndex;
      return node;
    }
    return null;
  }

  getNextMove() {
    if (!this.currentNode) return null;

    const currentIndex = this.getNodeIndex(this.currentNode);
    if (currentIndex !== -1 && currentIndex < this.mainline.length - 1) {
      return this.mainline[currentIndex + 1];
    }

    return this.currentNode.children[0] || null;
  }

  getPreviousMove() {
    if (this.currentNode === this.mainline[0]) return null;

    const currentIndex = this.getNodeIndex(this.currentNode);
    if (currentIndex > 0) return this.mainline[currentIndex - 1];

    return this.currentNode.parentId
      ? this.nodeMap.get(this.currentNode.parentId)
      : null;
  }

  getPathToNode(nodeId) {
    const node = this.nodeMap.get(nodeId);
    if (!node) return [];

    const nodeIndex = this.getNodeIndex(node);
    if (nodeIndex !== -1) {
      return this.mainline.slice(1, nodeIndex + 1);
    }

    const path = [];
    let current = node;
    while (current && current !== this.mainline[0]) {
      path.unshift(current);
      current = current.parentId ? this.nodeMap.get(current.parentId) : null;
    }

    return path;
  }

  getFinalMove() {
    return this.final;
  }

  /**
   * Gets an array of moves (with san notation) from root to the specified node
   * @param {string} nodeId - The target node ID
   * @returns {Array} Array of move objects with san property
   */
  getMovesToNode(nodeId) {
    const path = this.getPathToNode(nodeId);
    return path
      .filter((node) => node.move && node.san)
      .map((node) => ({ san: node.san, move: node.move }));
  }

  render(containerId, clickHandler, showClassification = true) {
    const $container = $(`#${containerId}`);
    if (!$container.length) return;

    $container.empty();
    const $mainLine = $("<div>").addClass("main-line").appendTo($container);

    let currentTurnContainer = null;
    const rootHasVariations = this.mainline[0].children.length > 0;

    for (let i = 1; i < this.mainline.length; i++) {
      const node = this.mainline[i];

      if (node.move?.color === "w" || !currentTurnContainer) {
        currentTurnContainer = $("<div>").addClass("turn").appendTo($mainLine);
      }

      const moveElement = this._createMoveElement(node, clickHandler, showClassification);
      currentTurnContainer.append(moveElement);

      if ((i === 1 && rootHasVariations) || node.children.length > 0) {
        const $variationsContainer = $("<div>")
          .addClass("variations")
          .appendTo($mainLine);

        (i === 1 && rootHasVariations
          ? this.mainline[0].children
          : node.children
        ).forEach((child) => {
          const $variationContainer = $("<div>")
            .addClass("variation")
            .appendTo($variationsContainer);
          this._renderVariation(child, $variationContainer[0], clickHandler, showClassification);
        });
      }
    }
  }

  _renderVariation(node, container, clickHandler, showClassification = true) {
    const movesInVariation = this._buildVariationMovesList(node);
    let currentTurnContainer = null;

    for (let i = 0; i < movesInVariation.length; i++) {
      const currentNode = movesInVariation[i];

      if (i === 0 && currentNode.move?.color === "b") {
        const $turnContainer = $("<div>").addClass("turn").appendTo(container);
        const $parentElement = $("<div>").addClass("move-entry");

        $parentElement
          .append(
            $("<span>")
              .addClass("move-number")
              .text(Math.floor(currentNode.moveNumber) + "... "),
          )
          .append(this._createMoveContainer(currentNode, clickHandler, showClassification));

        $turnContainer.append($parentElement);
        currentTurnContainer = $turnContainer;
      } else if (currentNode.move?.color === "w" || !currentTurnContainer) {
        currentTurnContainer = $("<div>").addClass("turn").appendTo(container);
        currentTurnContainer.append(
          this._createMoveElement(currentNode, clickHandler, showClassification),
        );
      } else {
        currentTurnContainer.append(
          this._createMoveElement(currentNode, clickHandler, showClassification),
        );
      }

      if (currentNode.children.length > 1) {
        const $variationsContainer = $("<div>")
          .addClass("variations")
          .appendTo(container);

        for (let j = 1; j < currentNode.children.length; j++) {
          const $variationContainer = $("<div>")
            .addClass("variation")
            .appendTo($variationsContainer);
          this._renderVariation(
            currentNode.children[j],
            $variationContainer[0],
            clickHandler,
            showClassification,
          );
        }
      }
    }
  }

  _buildVariationMovesList(startNode) {
    const movesList = [startNode];
    let currentNode = startNode;

    while (currentNode.children?.length > 0) {
      currentNode = currentNode.children[0];
      movesList.push(currentNode);
    }

    return movesList;
  }

  _createMoveElement(node, clickHandler, showClassification = true) {
    const $parentElement = $("<div>").addClass("move-entry");

    if (node.move?.color === "w" && node.moveNumber) {
      $parentElement.append(
        $("<span>")
          .addClass("move-number")
          .text(Math.floor(node.moveNumber) + "."),
      );
    }

    $parentElement.append(this._createMoveContainer(node, clickHandler, showClassification));

    return $parentElement[0];
  }

  _createMoveContainer(node, clickHandler, showClassification = true) {
    const $moveContainer = $("<div>").addClass("move-container");

    if (node.classification && showClassification) {
      $moveContainer.addClass(`${node.classification}-container`);
    }
    if (node.evaluationStatus) {
      $moveContainer.addClass(`evaluation-${node.evaluationStatus}`);
    }

    // Only show loading icon when showClassification is true
    if (node.evaluationStatus === "pending" && showClassification) {
      $moveContainer.append(
        $("<div>")
          .append(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><path d="M463.5 224l8.5 0c13.3 0 24-10.7 24-24l0-128c0-9.7-5.8-18.5-14.8-22.2s-19.3-1.7-26.2 5.2L413.4 96.6c-87.6-86.5-228.7-86.2-315.8 1c-87.5 87.5-87.5 229.3 0 316.8s229.3 87.5 316.8 0c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0c-62.5 62.5-163.8 62.5-226.3 0s-62.5-163.8 0-226.3c62.2-62.2 162.7-62.5 225.3-1L327 183c-6.9 6.9-8.9 17.2-5.2 26.2s12.5 14.8 22.2 14.8l119.5 0z" fill="currentColor"/></svg>',
          )
          .addClass("loading-icon"),
      );
    } else if (node.classification && showClassification) {
      const classification = Classification[node.classification.toUpperCase()];

      $moveContainer.append(
        classification?.cachedImg
          ? classification.cachedImg.cloneNode(true)
          : $("<img>")
              .addClass("move-icon")
              .attr("src", classification.src)
              .attr("alt", node.classification),
      );
    }

    const $moveText = $("<span>")
      .addClass("move")
      .text(node.san)
      .attr("data-node-id", node.id);

    if (node.classification && showClassification) {
      const classification = Classification[node.classification.toUpperCase()];
      if (classification) {
        $moveText.addClass(`${classification.type}-move`);
      }
    }

    $moveContainer.append($moveText);

    if (clickHandler) {
      $moveContainer.on("click", () => clickHandler(node));
    }

    return $moveContainer;
  }

  updateCurrentMove(nodeId) {
    $(".current-move").removeClass("current-move");
    $(".current-container").removeClass("current-container");

    const $moveElement = $(`[data-node-id="${nodeId}"]`);
    if ($moveElement.length > 0) {
      $moveElement.addClass("current-move");
      $moveElement.closest(".move-container").addClass("current-container");
    }
  }

  updateNodeClassification(node, board, showClassification = true) {
    if (node !== this.mainline[0] && node.move && node.classification) {
      const from = board.algebraicToIndex(node.move.from, board.flipped);
      const to = board.algebraicToIndex(node.move.to, board.flipped);

      board.addClassification(
        node.classification,
        board.getSquare(from, board.flipped),
        board.getSquare(to, board.flipped),
        showClassification,
      );
    }
  }

  destroy() {
    const $container = $("#move-tree");
    $container.off();
    $container.empty();

    this.mainline = [];
    this.nodeMap?.clear();
    this.nodeMap = new Map();
    this.currentNode = null;
    this.final = null;
    this.idCounter = 0;
    this.currentIndex = 0;
    this.clockData?.clear?.();
    this.clockData = new Map();
  }
}
