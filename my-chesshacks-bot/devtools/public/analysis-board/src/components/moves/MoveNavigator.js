import { MoveInformation } from "./MoveInformation.js";
import { EvaluationBar } from "../board/EvaluationBar.js";
import { EngineLines } from "./EngineLines.js";
import { GameGraph } from "../report/GameGraph.js";
import { Clock } from "../board/Clock.js";
import { DebugInformation } from "../debug/DebugInformation.js";

export class MoveNavigator {
  constructor(chessUI) {
    this.chessUI = chessUI;

    this.boundHandlers = {
      forward: () => this.handleForwardMove(),
      backward: () => this.handleBackwardMove(),
      flip: () => this.handleFlipBoard(),
      restart: () => this.handleRestart(),
      skip: () => this.handleSkipToEnd(),
      copyFen: () => this.handleCopyFenToClipboard(),
      quickMenuToggle: (e) => this.handleQuickMenuToggle(e),
      shareFen: () => this.handleShareFen(),
      copyPgn: () => this.handleCopyPgn(),
      flipBoardQuick: () => this.handleFlipBoard(),
      downloadPgn: () => this.handleDownloadPgn(),
      documentClick: (e) => this.handleDocumentClick(e),
      documentKeydown: (e) => this.handleDocumentKeydown(e),
      boardUserMove: (moveObj) => this.handleUserMove(moveObj)
    };
  }

  /**
   * Sets up event handlers for navigation buttons
   */
  setupEventHandlers() {
    $("#forward").on("click", this.boundHandlers.forward);
    $("#backward").on("click", this.boundHandlers.backward);
    $("#flip").on("click", this.boundHandlers.flip);
    $("#restart").on("click", this.boundHandlers.restart);
    $("#skip-to-end").on("click", this.boundHandlers.skip);
    $("#fen-to-clipboard").on("click", this.boundHandlers.copyFen);

    // Quick menu handlers
    $("#popup-quick-menu").on("click", this.boundHandlers.quickMenuToggle);
    $("#share-fen").on("click", this.boundHandlers.shareFen);
    $("#copy-pgn").on("click", this.boundHandlers.copyPgn);
    $("#flip-board").on("click", this.boundHandlers.flipBoardQuick);
    $("#download-pgn").on("click", this.boundHandlers.downloadPgn);

    // Close quick menu when clicking outside
    $(document).on("click", this.boundHandlers.documentClick);

    // Keyboard navigation
    $(document).on("keydown", this.boundHandlers.documentKeydown);

    // Set up new chessboard event listeners
    this.chessUI.board.on("usermove", this.boundHandlers.boardUserMove);
  }

  handleCopyFenToClipboard() {
    const currentNode = this.chessUI.moveTree.currentNode;
    const fen = currentNode.move.after;
    navigator.clipboard.writeText(fen);
  }

  updateAfterMove(node) {
    if (this.chessUI?._isDestroyed) return;

    this.chessUI.moveTree.updateCurrentMove(node.id);

    const showClassification =
      this.chessUI.settingsMenu.getSettingValue("showClassification");

    // Show/hide evaluation bar based on setting
    const $evalBar = $(".eval-bar");
    if (showClassification) {
      $evalBar.show();
      if (node.evalScore !== undefined) {
        EvaluationBar.updateEvaluationBar(node);
      }
    } else {
      $evalBar.hide();
    }

    const prevNode = this.chessUI.moveTree.getPreviousMove();
    MoveInformation.updateMoveInfo(node, prevNode, showClassification);

    EngineLines.updateEngineLines(
      node,
      this.chessUI.moveTree,
      (node) => this.handleTreeNodeClick(node),
      (node, resultFen, prevFen) =>
        this.queueMoveForEvaluation(node, resultFen, prevFen),
      showClassification
    );

    GameGraph.updateCurrentMoveNumber(node.moveNumber);

    // Only show best move arrow if both the arrow setting and classification setting are enabled
    if (
      showClassification &&
      this.chessUI.settingsMenu.getSettingValue("showBestMoveArrow")
    ) {
      if (prevNode?.evaluatedMove?.lines?.length > 0) {
        const bestLine = prevNode.evaluatedMove.lines.find(
          (line) => line.id === 1
        );
        if (bestLine) {
          const bestMoveUci =
            bestLine.uciMove ||
            (bestLine.pv && bestLine.pv.length > 0 ? bestLine.pv[0] : null);
          if (bestMoveUci) {
            const fromIdx = this.chessUI.board.algebraicToIndex(
              bestMoveUci.substring(0, 2),
              this.chessUI.board.flipped
            );
            const toIdx = this.chessUI.board.algebraicToIndex(
              bestMoveUci.substring(2, 4),
              this.chessUI.board.flipped
            );
            this.chessUI.board.createArrow(
              this.chessUI.board.getSquare(fromIdx),
              this.chessUI.board.getSquare(toIdx)
            );
          }
        }
      }
    }

    // Update debug information (including probability arrows)
    const showProbabilityArrows = this.chessUI.settingsMenu.getSettingValue(
      "showProbabilityArrows"
    );
    DebugInformation.updateDebugInfo(
      node,
      prevNode,
      this.chessUI.board,
      showProbabilityArrows
    );

    // Update clocks
    Clock.updateFromMoveTree(
      this.chessUI.moveTree,
      this.chessUI.board.flipped,
      this.chessUI.game?.pgn
    );

    if (typeof this.chessUI._updateBoardInteractivity === "function") {
      this.chessUI._updateBoardInteractivity();
    }
  }

  handleForwardMove() {
    if (this.chessUI?._isDestroyed) return;

    const nextNode = this.chessUI.moveTree.getNextMove();
    if (!nextNode || !nextNode.move) return;

    const showClassification =
      this.chessUI.settingsMenu.getSettingValue("showClassification");
    const classification = nextNode.classification;
    this.chessUI.board.move(
      nextNode.move,
      true,
      showClassification ? classification : null,
      nextNode.move.before,
      false,
      nextNode.move.promotion,
      false
    );

    this.chessUI.moveTree.navigateTo(nextNode.id);
    this.updateAfterMove(nextNode);
  }

  handleBackwardMove() {
    if (this.chessUI?._isDestroyed) return;

    if (this.chessUI.moveTree.currentNode === this.chessUI.moveTree.mainline[0])
      return;

    const currentNode = this.chessUI.moveTree.currentNode;
    const prevNode = this.chessUI.moveTree.getPreviousMove();

    if (!prevNode || !currentNode.move) return;

    this.chessUI.board.unmove(true, currentNode.move, currentNode.move.before);

    const showClassification =
      this.chessUI.settingsMenu.getSettingValue("showClassification");
    this.chessUI.moveTree.navigateTo(prevNode.id);
    this.chessUI.moveTree.updateNodeClassification(
      prevNode,
      this.chessUI.board,
      showClassification
    );

    this.updateAfterMove(prevNode);
  }

  handleFlipBoard() {
    if (this.chessUI?._isDestroyed) return;

    this.chessUI.board.flip();

    // Toggle the flipped class on the progress bar
    $(".eval-bar").toggleClass("flipped");
    $(".chess-container").toggleClass("flipped");

    const showClassification =
      this.chessUI.settingsMenu.getSettingValue("showClassification");
    const currentNode = this.chessUI.moveTree.currentNode;
    this.chessUI.moveTree.updateNodeClassification(
      currentNode,
      this.chessUI.board,
      showClassification
    );

    if (currentNode.evalScore !== undefined) {
      EvaluationBar.updateEvaluationBar(currentNode);
    }

    // Update clocks after flipping
    Clock.updateFromMoveTree(
      this.chessUI.moveTree,
      this.chessUI.board.flipped,
      this.chessUI.game?.pgn
    );

    if (typeof this.chessUI._updateBoardInteractivity === "function") {
      this.chessUI._updateBoardInteractivity();
    }
  }

  handleRestart() {
    if (this.chessUI?._isDestroyed) return;

    this.chessUI.board.fen();

    this.chessUI.moveTree.navigateTo("root");
    this.chessUI.moveTree.updateCurrentMove("root");

    const showClassification =
      this.chessUI.settingsMenu.getSettingValue("showClassification");
    GameGraph.updateCurrentMoveNumber(0);

    // Show/hide evaluation bar based on setting
    const $evalBar = $(".eval-bar");
    if (showClassification) {
      $evalBar.show();
      EvaluationBar.updateEvaluationBar();
    } else {
      $evalBar.hide();
    }

    EngineLines.updateEngineLines(
      this.chessUI.moveTree.currentNode,
      this.chessUI.moveTree,
      (node) => this.handleTreeNodeClick(node),
      (node, resultFen, prevFen) =>
        this.queueMoveForEvaluation(node, resultFen, prevFen),
      showClassification
    );

    // Update debug information
    DebugInformation.updateDebugInfo(
      this.chessUI.moveTree.currentNode,
      this.chessUI.moveTree.currentNode
    );

    // Update clocks for starting position
    Clock.updateFromMoveTree(
      this.chessUI.moveTree,
      this.chessUI.board.flipped,
      this.chessUI.game?.pgn
    );

    if (typeof this.chessUI._updateBoardInteractivity === "function") {
      this.chessUI._updateBoardInteractivity();
    }
  }

  handleSkipToEnd() {
    if (this.chessUI?._isDestroyed) return;

    const lastMove = this.chessUI.moveTree.getFinalMove();
    this.handleTreeNodeClick(lastMove);
  }

  handleUserMove(moveObj) {
    if (this.chessUI?._isDestroyed) return;

    // Check if the move exists in the mainline next
    const currentIndex = this.chessUI.moveTree.getNodeIndex(
      this.chessUI.moveTree.currentNode
    );
    if (
      currentIndex !== -1 &&
      currentIndex + 1 < this.chessUI.moveTree.mainline.length
    ) {
      const nextMainlineMove = this.chessUI.moveTree.mainline[currentIndex + 1];
      if (
        nextMainlineMove.move &&
        nextMainlineMove.move.from === moveObj.from &&
        nextMainlineMove.move.to === moveObj.to &&
        nextMainlineMove.move.promotion === moveObj.promotion
      ) {
        return this.navigateToExistingMove(nextMainlineMove);
      }
    }

    // Check if the move already exists as a variation
    const existingChild = this.chessUI.moveTree.currentNode.children.find(
      (child) =>
        child.move &&
        child.move.from === moveObj.from &&
        child.move.to === moveObj.to &&
        child.move.promotion === moveObj.promotion
    );

    return existingChild
      ? this.navigateToExistingMove(existingChild)
      : this.createNewVariation(moveObj);
  }

  handleTreeNodeClick(node) {
    if (this.chessUI?._isDestroyed) return;

    // Handle path navigation
    const path = this.chessUI.moveTree.getPathToNode(node.id);

    if (path.length > 0) {
      const lastNode = path[path.length - 1];

      // Already at this node - do nothing
      if (lastNode.id === this.chessUI.moveTree.currentNode.id) {
        return;
      }

      // Check if one move ahead
      if (
        this.chessUI.moveTree.currentNode.children.some(
          (child) => child.id === lastNode.id
        )
      ) {
        const nextNode = this.chessUI.moveTree.getNextMove();
        if (nextNode && nextNode.id === lastNode.id) {
          $("#forward").trigger("click");
          return;
        }
      }

      // Check if one move behind (parent)
      const currentIndex = this.chessUI.moveTree.getNodeIndex(
        this.chessUI.moveTree.currentNode
      );
      const lastNodeIndex = this.chessUI.moveTree.getNodeIndex(lastNode);
      if (currentIndex > 0 && lastNodeIndex === currentIndex - 1) {
        $("#backward").trigger("click");
        return;
      }

      // Check if parent of variation
      if (this.chessUI.moveTree.currentNode.parentId === lastNode.id) {
        $("#backward").trigger("click");
        return;
      }
    }

    // For moves more than one step away
    const targetNode = this.chessUI.moveTree.nodeMap.get(node.id);

    //console.log(targetNode.evaluatedMove?.comment)

    // Handle root node case
    if (!targetNode || targetNode.id === "root") {
      this.handleRestart();
      return;
    }

    // Handle multi-step navigation
    this.navigateToTargetPosition(targetNode);
  }

  navigateToExistingMove(moveNode) {
    if (this.chessUI?._isDestroyed) return true;

    this.chessUI.moveTree.navigateTo(moveNode.id);

    const showClassification =
      this.chessUI.settingsMenu.getSettingValue("showClassification");
    // Add classification to the board
    const fromIdx = this.chessUI.board.algebraicToIndex(
      moveNode.move.from,
      this.chessUI.board.flipped
    );
    const toIdx = this.chessUI.board.algebraicToIndex(
      moveNode.move.to,
      this.chessUI.board.flipped
    );
    this.chessUI.board.addClassification(
      moveNode.classification,
      this.chessUI.board.getSquare(fromIdx, this.chessUI.board.flipped),
      this.chessUI.board.getSquare(toIdx, this.chessUI.board.flipped),
      showClassification
    );

    this.updateAfterMove(moveNode);

    return true;
  }

  navigateToTargetPosition(targetNode) {
    if (this.chessUI?._isDestroyed) return;

    let parentNode;

    // Find the parent node based on whether the target is in mainline or a variation
    const targetNodeIndex = this.chessUI.moveTree.getNodeIndex(targetNode);
    if (targetNodeIndex !== -1) {
      // Target is in mainline
      parentNode = this.chessUI.moveTree.mainline[targetNodeIndex - 1];
    } else if (targetNode.parentId) {
      // Target is a variation
      parentNode = this.chessUI.moveTree.nodeMap.get(targetNode.parentId);
    } else {
      console.error("Cannot find parent for node", targetNode);
      return;
    }

    const showClassification =
      this.chessUI.settingsMenu.getSettingValue("showClassification");
    this.chessUI.board.fen(targetNode.move.before);
    if (targetNode.move) {
      this.chessUI.board.move(
        targetNode.move,
        true,
        showClassification ? targetNode.classification : null,
        targetNode.move.before,
        false,
        targetNode.move.promotion
      );
    }

    // Update move tree and UI
    this.chessUI.moveTree.navigateTo(targetNode.id);
    this.updateAfterMove(targetNode);
  }

  createNewVariation(moveObj) {
    if (this.chessUI?._isDestroyed) return false;

    // Add the move to the tree
    const newNode = this.chessUI.moveTree.addMove(
      moveObj,
      this.chessUI.moveTree.currentNode.id
    );
    newNode.evaluationStatus = "pending";

    const showClassification =
      this.chessUI.settingsMenu.getSettingValue("showClassification");
    this.chessUI.moveTree.navigateTo(newNode.id);
    this.chessUI.moveTree.render(
      "move-tree",
      (node) => this.handleTreeNodeClick(node),
      showClassification
    );
    this.updateAfterMove(newNode);

    // Queue for evaluation
    const fenBefore = newNode.move.before;
    const fenAfter = newNode.move.after;
    this.queueMoveForEvaluation(newNode, fenAfter, fenBefore);

    return true;
  }

  queueMoveForEvaluation(node, resultFen, prevFen) {
    if (this.chessUI?._isDestroyed) return;

    this.chessUI.evaluationQueue.addToQueue(
      node,
      resultFen,
      prevFen,
      (evaluatedMove) => {
        if (this.chessUI?._isDestroyed) return;

        // Update classification and evaluation data
        this.chessUI.moveTree.updateClassification(node.id, evaluatedMove);
        node.evaluationStatus = "complete";

        const topLine = evaluatedMove.lines.find((line) => line.id === 1);
        if (topLine) {
          node.evalScore = topLine.score;
          node.evalType = topLine.type || "cp";
        }

        const showClassification =
          this.chessUI.settingsMenu.getSettingValue("showClassification");
        // Update UI if this is the current node
        const currentNode = this.chessUI.moveTree.currentNode;
        if (currentNode.id === node.id && node.move) {
          const fromIdx = this.chessUI.board.algebraicToIndex(
            node.move.from,
            this.chessUI.board.flipped
          );
          const toIdx = this.chessUI.board.algebraicToIndex(
            node.move.to,
            this.chessUI.board.flipped
          );

          this.chessUI.board.addClassification(
            evaluatedMove.classification.type,
            this.chessUI.board.getSquare(fromIdx, this.chessUI.board.flipped),
            this.chessUI.board.getSquare(toIdx, this.chessUI.board.flipped),
            showClassification
          );
        }

        this.chessUI.moveTree.render(
          "move-tree",
          (node) => this.handleTreeNodeClick(node),
          showClassification
        );
        this.updateAfterMove(currentNode);
      },
      this.chessUI.moveTree
    );
  }

  handleQuickMenuToggle(e) {
    e.stopPropagation();
    if (this.chessUI?._isDestroyed) return;

    const menu = $("#quick-menu");
    const isVisible = menu.hasClass("show");

    if (isVisible) {
      menu.removeClass("show");
    } else {
      menu.addClass("show");
    }
  }

  handleDocumentClick(e) {
    if (this.chessUI?._isDestroyed) return;

    const menu = $("#quick-menu");
    const button = $("#play");

    if (
      !menu.is(e.target) &&
      menu.has(e.target).length === 0 &&
      !button.is(e.target) &&
      button.has(e.target).length === 0
    ) {
      menu.removeClass("show");
    }
  }

  handleDocumentKeydown(e) {
    if (this.chessUI?._isDestroyed) return;

    switch (e.keyCode) {
      case 39: // Right arrow
        e.preventDefault();
        this.handleForwardMove();
        break;
      case 37: // Left arrow
        e.preventDefault();
        this.handleBackwardMove();
        break;
      case 38: // Up arrow
        e.preventDefault();
        this.handleRestart();
        break;
      case 70: // F key
        this.handleFlipBoard();
        break;
      default:
        break;
    }
  }

  handleShareFen() {
    if (this.chessUI?._isDestroyed) return;

    const currentNode = this.chessUI.moveTree.currentNode;
    const fen = currentNode.move ? currentNode.move.after : currentNode.fen;

    if (navigator.share) {
      navigator
        .share({
          title: "Centichess Analysis",
          text: `Check out this game: ${fen}`,
          url: window.location.href
        })
        .catch(console.error);
    } else {
      // Fallback to clipboard
      navigator.clipboard
        .writeText(fen)
        .then(() => {
          this.showNotification("Game URL copied to clipboard!");
        })
        .catch(() => {
          this.showNotification("Failed to copy game URL.");
        });
    }
    $("#quick-menu").removeClass("show");
  }

  handleCopyPgn() {
    if (this.chessUI?._isDestroyed) return;

    const pgn =
      (this.chessUI && typeof this.chessUI.getCurrentPgn === "function"
        ? this.chessUI.getCurrentPgn()
        : null) ||
      this.chessUI.game?.pgn ||
      "";

    if (pgn) {
      navigator.clipboard
        .writeText(pgn)
        .then(() => {
          this.showNotification("PGN copied to clipboard!");
        })
        .catch(() => {
          this.showNotification("Failed to copy PGN");
        });
    } else {
      this.showNotification("No PGN available");
    }
    $("#quick-menu").removeClass("show");
  }

  handleDownloadPgn() {
    if (this.chessUI?._isDestroyed) return;

    const pgn =
      (this.chessUI && typeof this.chessUI.getCurrentPgn === "function"
        ? this.chessUI.getCurrentPgn()
        : null) ||
      this.chessUI.game?.pgn ||
      "";

    if (pgn) {
      const blob = new Blob([pgn], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `chess-game-${new Date().toISOString().split("T")[0]}.pgn`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      this.showNotification("PGN downloaded!");
    } else {
      this.showNotification("No PGN available");
    }
    $("#quick-menu").removeClass("show");
  }

  showNotification(message) {
    if (this.chessUI?._isDestroyed) return;

    // Create a simple notification
    const notification = $(`
            <div class="quick-notification" style="
                position: fixed;
                top: 20px;
                right: 20px;
                background: var(--sidebar-base);
                color: var(--text-primary);
                padding: 12px 20px;
                border-radius: 8px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
                z-index: 10000;
                font-size: 14px;
                border: 1px solid var(--dark-border);
                animation: slideInRight 0.3s ease;
            ">${message}</div>
        `);

    $("body").append(notification);

    setTimeout(() => {
      notification.fadeOut(300, () => notification.remove());
    }, 3000);
  }

  destroy() {
    const handlers = this.boundHandlers;
    if (!handlers) return;

    $("#forward").off("click", handlers.forward);
    $("#backward").off("click", handlers.backward);
    $("#flip").off("click", handlers.flip);
    $("#restart").off("click", handlers.restart);
    $("#skip-to-end").off("click", handlers.skip);
    $("#fen-to-clipboard").off("click", handlers.copyFen);

    $("#popup-quick-menu").off("click", handlers.quickMenuToggle);
    $("#share-fen").off("click", handlers.shareFen);
    $("#copy-pgn").off("click", handlers.copyPgn);
    $("#flip-board").off("click", handlers.flipBoardQuick);
    $("#download-pgn").off("click", handlers.downloadPgn);

    $(document).off("click", handlers.documentClick);
    $(document).off("keydown", handlers.documentKeydown);

    if (this.chessUI?.board) {
      this.chessUI.board.off("usermove", handlers.boardUserMove);
    }

    $("#quick-menu").removeClass("show");
    this.boundHandlers = null;
  }
}
