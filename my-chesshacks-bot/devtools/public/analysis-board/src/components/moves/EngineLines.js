import { Chess } from "../../../libs/chess.js";
import { Classification } from "../../classification/MoveClassifier.js";
import { MoveEvaluator } from "../../evaluation/MoveEvaluator.js";
import { EvaluationBar } from "../board/EvaluationBar.js";

export const IgnoredSuggestionTypes = [
    Classification.BRILLIANT.type,
    Classification.GREAT.type,
    Classification.PERFECT.type,
    Classification.THEORY.type,
    Classification.FORCED.type
];

export class EngineLines {
    static updateEngineLines(node, moveTree, handleTreeNodeClick, queueMoveForEvaluation, showClassification = true) {
        const $engineLines = $(".engine-lines").empty();

        // Hide the panel completely if showClassification is false
        if (!showClassification) {
            $engineLines.hide();
            return;
        }

        // Show the panel if it was hidden
        $engineLines.show();

        // Create title element
        const titleElement = this.createEngineTitleElement(node);
        $engineLines.append(titleElement);

        // Handle missing node
        if (!node) {
            this.showEngineWaitingMessage("Select a position to view analysis");
            return;
        }

        // Handle root node with pre-computed evaluation
        if (node.id === 'root') {
            node.evaluatedMove = MoveEvaluator.startPositionEvaluation;
            node.fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        }

        // Handle game-over position
        const chess = new Chess(node.fen);
        if (chess.isGameOver()) {
            this.showGameOverMessage(chess);
            return;
        }

        // Handle missing evaluation data
        if (!node.evaluatedMove || !node.evaluatedMove.lines) {
            this.showEngineWaitingMessage("Analyzing position...", true);
            return;
        }

        // Display evaluation lines
        this.displayEngineLines(node, moveTree, handleTreeNodeClick, queueMoveForEvaluation);
    }

    static displayEngineLines(node, moveTree, handleTreeNodeClick, queueMoveForEvaluation) {
        const lines = node.evaluatedMove.lines;
        const $linesContainer = $("<div>").addClass("engine-lines-container");
        const sortedLines = [...lines].sort((a, b) => a.id - b.id);

        // 3 lines only
        for (let i = 0; i < Math.min(3, sortedLines.length); i++) {
            const line = sortedLines[i];
            if (!line) continue;

            const $lineContainer = this.createLineContainer(node, line, i, moveTree, handleTreeNodeClick, queueMoveForEvaluation);
            $linesContainer.append($lineContainer);
        }

        $(".engine-lines").append($linesContainer);
    }

    static showGameOverMessage(chess) {
        const $resultContainer = $("<div>").addClass("engine-lines-container");
        const $resultLine = $("<div>").addClass("engine-line");
        const $resultBox = $("<div>").addClass("game-result");

        let resultText = "";
        const isMate = chess.isCheckmate()
        if (isMate) {
            const turn = chess.turn() === 'w';
            resultText = (turn ? "Black" : "White") + " won by checkmate";
            if (turn) $resultBox.addClass("black");
        } else if (chess.isStalemate()) {
            resultText = "Draw by stalemate";
        } else if (chess.isInsufficientMaterial()) {
            resultText = "Draw by insufficient material";
        } else if (chess.isThreefoldRepetition()) {
            resultText = "Draw by threefold repetition";
        } else if (chess.isDraw()) {
            resultText = "Draw by 50-move rule";
        }

        $resultBox.text(resultText);
        $resultLine.append($resultBox);
        $resultContainer.append($resultLine);
        $(".engine-lines").append($resultContainer);

        // Update evaluation bar based on result
        const evalObj = {
            evalScore: -0,
            evalType: isMate ? "mate" : "cp",
            mateForBlack: chess.turn() === 'w' && isMate
        };
        EvaluationBar.updateEvaluationBar(evalObj);
    }

    static createLineContainer(node, line, lineIndex, moveTree, handleTreeNodeClick, queueMoveForEvaluation) {
        const $lineContainer = $("<div>").addClass("engine-line");

        const $scoreBox = this.createScoreBox(line);
        $lineContainer.append($scoreBox);

        const $movesContainer = $("<div>").addClass("engine-moves");

        if (line.pv?.length > 0) {
            const { movesList, uciMovesList } = this.parsePrincipalVariation(node, line);
            this.createMovesContent($movesContainer, movesList, uciMovesList, node, line, lineIndex, moveTree, handleTreeNodeClick, queueMoveForEvaluation);

            $lineContainer.attr({
                "data-moves": JSON.stringify(movesList.map(m => m.moveObj)),
                "data-uci-moves": JSON.stringify(uciMovesList)
            });
        }

        $lineContainer.append($movesContainer);
        return $lineContainer;
    }

    static createScoreBox(line) {
        let scoreText;

        if (line.type === "mate") {
            scoreText = (line.score > 0) ? "M" + line.score : "M" + Math.abs(line.score);
        } else {
            let evalValue = line.score / 100;
            scoreText = evalValue > 0 ? "+" + evalValue.toFixed(2) : evalValue.toFixed(2);
        }

        return $("<div>")
            .addClass("engine-score")
            .addClass(line.score >= 0 ? "white-score" : "black-score")
            .text(scoreText);
    }

    static parsePrincipalVariation(node, line) {
        const tempChess = new Chess(node.fen);
        const movesList = [];
        const uciMovesList = [];

        let moveNumber = Math.ceil(node.moveNumber);
        let isWhiteTurn = tempChess.turn() === 'w';

        // Parse moves from UCI to SAN format
        line.pv.forEach(uciMove => {
            try {
                uciMovesList.push(uciMove);

                const from = uciMove.substring(0, 2);
                const to = uciMove.substring(2, 4);
                const promotion = uciMove.length > 4 ? uciMove.substring(4, 5) : undefined;

                const move = tempChess.move({ from, to, promotion });
                //console.log(move);
                if (move) {
                    // Create move prefix (number + dots)
                    let prefix = '';
                    if (isWhiteTurn) {
                        prefix = moveNumber + '. ';
                    } else if (movesList.length === 0) {
                        prefix = moveNumber + '... ';
                    }

                    movesList.push({
                        text: prefix + move.san,
                        uci: uciMove,
                        moveObj: { from, to, promotion },
                        index: movesList.length
                    });

                    // Update move number and turn
                    if (!isWhiteTurn) moveNumber++;
                    isWhiteTurn = !isWhiteTurn;
                }
            } catch (e) { 
                console.warn("Failed to parse UCI move:", uciMove, e);
            }
        });

        return { movesList, uciMovesList };
    }

    static createMovesContent($movesContainer, movesList, uciMovesList, node, line, lineIndex, moveTree, handleTreeNodeClick, queueMoveForEvaluation) {
        const maxVisibleMoves = 3;
        const $movesContentContainer = $("<div>").addClass("moves-content");

        if (movesList.length > maxVisibleMoves) {
            // Create container for visible moves
            const visibleMovesList = movesList.slice(0, maxVisibleMoves);
            const hiddenMovesList = movesList.slice(maxVisibleMoves);

            // Add visible moves
            this.createVisibleMovesElements(visibleMovesList, $movesContentContainer, node, line, lineIndex, moveTree, handleTreeNodeClick, queueMoveForEvaluation);
            $movesContainer.append($movesContentContainer);

            // Create expand button
            const $expandButton = $("<span>")
                .addClass("expand-button")
                .html('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><!--!Font Awesome Free 6.7.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2025 Fonticons, Inc.--><path d="M233.4 406.6c12.5 12.5 32.8 12.5 45.3 0l192-192c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0L256 338.7 86.6 169.4c-12.5-12.5-32.8-12.5-45.3 0s-12.5 32.8 0 45.3l192 192z" fill="currentColor"/></svg>');
            $movesContainer.append($expandButton);

            // Create container for hidden moves
            const $hiddenMovesEl = this.createHiddenMovesElements(hiddenMovesList, node, line, lineIndex, maxVisibleMoves, moveTree, handleTreeNodeClick, queueMoveForEvaluation);
            $movesContentContainer.append($hiddenMovesEl);

            // Add expand/collapse functionality
            this.setupExpandCollapseHandler($expandButton, $hiddenMovesEl);
        } else {
            // If no truncation needed, just show all moves
            this.createSimpleMovesElements(movesList, $movesContentContainer, node, line, lineIndex, moveTree, handleTreeNodeClick, queueMoveForEvaluation);
            $movesContainer.append($movesContentContainer);
        }
    }

    static createVisibleMovesElements(visibleMovesList, $container, node, line, lineIndex, moveTree, handleTreeNodeClick, queueMoveForEvaluation) {
        const $visibleMovesEl = $("<span>").addClass("visible-moves");

        visibleMovesList.forEach((moveData, idx) => {
            const $moveEl = $("<span>")
                .addClass("clickable-move")
                .text(moveData.text)
                .attr({
                    "data-uci": moveData.uci,
                    "data-position": idx,
                    "data-line": lineIndex
                })
                .on("click", () => {
                    this.handleEngineLineClick(node, line, idx + 1, moveTree, handleTreeNodeClick, queueMoveForEvaluation);
                });

            $visibleMovesEl.append($moveEl);
            if (idx < visibleMovesList.length - 1) {
                $visibleMovesEl.append(" ");
            }
        });

        $container.append($visibleMovesEl);
    }

    static createHiddenMovesElements(hiddenMovesList, node, line, lineIndex, maxVisibleMoves, moveTree, handleTreeNodeClick, queueMoveForEvaluation) {
        const $hiddenMovesEl = $("<div>").addClass("hidden-moves");

        hiddenMovesList.forEach((moveData, idx) => {
            const $moveEl = $("<span>")
                .addClass("clickable-move")
                .text(moveData.text)
                .attr({
                    "data-uci": moveData.uci,
                    "data-position": idx + maxVisibleMoves,
                    "data-line": lineIndex
                })
                .on("click", () => {
                    this.handleEngineLineClick(node, line, idx + maxVisibleMoves + 1, moveTree, handleTreeNodeClick, queueMoveForEvaluation);
                });

            $hiddenMovesEl.append($moveEl);
            if (idx < hiddenMovesList.length - 1) {
                $hiddenMovesEl.append(" ");
            }
        });

        return $hiddenMovesEl;
    }

    static createSimpleMovesElements(movesList, $container, node, line, lineIndex, moveTree, handleTreeNodeClick, queueMoveForEvaluation) {
        movesList.forEach((moveData, idx) => {
            const $moveEl = $("<span>")
                .addClass("clickable-move")
                .text(moveData.text)
                .attr({
                    "data-uci": moveData.uci,
                    "data-position": idx,
                    "data-line": lineIndex
                })
                .on("click", () => {
                    this.handleEngineLineClick(node, line, idx + 1, moveTree, handleTreeNodeClick, queueMoveForEvaluation);
                });

            $container.append($moveEl);
            if (idx < movesList.length - 1) {
                $container.append(" ");
            }
        });
    }

    static setupExpandCollapseHandler($expandButton, $hiddenMovesEl) {
        $expandButton.on("click", function () {
            const isExpanded = $(this).hasClass("expanded");
            $(this).toggleClass("expanded", !isExpanded);

            $(this).html(isExpanded ?
                `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><!--!Font Awesome Free 6.7.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2025 Fonticons, Inc.--><path d="M233.4 406.6c12.5 12.5 32.8 12.5 45.3 0l192-192c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0L256 338.7 86.6 169.4c-12.5-12.5-32.8-12.5-45.3 0s-12.5 32.8 0 45.3l192 192z" fill="currentColor"/></svg>` :
                `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><!--!Font Awesome Free 6.7.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2025 Fonticons, Inc.--><path d="M233.4 105.4c12.5-12.5 32.8-12.5 45.3 0l192 192c12.5 12.5 12.5 32.8 0 45.3s-32.8 12.5-45.3 0L256 173.3 86.6 342.6c-12.5 12.5-32.8 12.5-45.3 0s-12.5-32.8 0-45.3l192-192z" fill="currentColor"/></svg>`);

            $hiddenMovesEl[isExpanded ? 'slideUp' : 'slideDown'](200);
        });
    }

    static createEngineTitleElement(node) {
        const hasEvaluation = (node?.evaluatedMove?.lines?.length > 0) || (node?.id === 'root');

        const $title = $("<div>")
            .addClass("section-title engine-lines-title")
            .addClass(hasEvaluation ? "has-evaluation" : "no-evaluation")
            .append($("<span>").text("Computer"));

        // Add depth info if available
        if (node?.evaluatedMove?.lines?.[0]) {
            const depth = node.evaluatedMove.lines[0].depth || "?";
            const engine = node.evaluatedMove.engine || "";
            $title.append(
                $("<span>").addClass("engine-depth")
                    .text("Depth " + depth + " " + engine)
            );
        }

        return $title;
    }

    static showEngineWaitingMessage(message, isLoading = false) {
        const $waitingMsg = $("<div>")
            .addClass("engine-lines-waiting")
            .append(
                $("<div>")
                    .append(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><!--!Font Awesome Free 6.7.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2025 Fonticons, Inc.--><path d="M463.5 224l8.5 0c13.3 0 24-10.7 24-24l0-128c0-9.7-5.8-18.5-14.8-22.2s-19.3-1.7-26.2 5.2L413.4 96.6c-87.6-86.5-228.7-86.2-315.8 1c-87.5 87.5-87.5 229.3 0 316.8s229.3 87.5 316.8 0c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0c-62.5 62.5-163.8 62.5-226.3 0s-62.5-163.8 0-226.3c62.2-62.2 162.7-62.5 225.3-1L327 183c-6.9 6.9-8.9 17.2-5.2 26.2s12.5 14.8 22.2 14.8l119.5 0z" fill="currentColor"/></svg>`)
                    .toggleClass("loading-icon", isLoading)
            )
            .append($("<span>").text(" " + message));
        
        $(".engine-lines").append($waitingMsg);
    }

    static handleEngineLineClick(currentNode, line, movesToPlay, moveTree, handleTreeNodeClick, queueMoveForEvaluation) {
        if (!line.pv?.length || movesToPlay <= 0 || movesToPlay > line.pv.length) {
            return;
        }

        const moves = line.pv.slice(0, movesToPlay);
        let chess = new Chess(currentNode.fen);
        let nodeToPlayFrom = currentNode;

        // Add each move as a variation
        for (let i = 0; i < moves.length; i++) {
            const uciMove = moves[i];
            const from = uciMove.substring(0, 2);
            const to = uciMove.substring(2, 4);
            const promotion = uciMove.length > 4 ? uciMove.substring(4, 5) : undefined;

            // Try to make the move
            const moveObj = chess.move({ from, to, promotion });
            console.log("engine line click", moveObj);
            if (!moveObj) return;

            // Check for existing node in mainline if nodeToPlayFrom is in mainline
            let existingNode = null;
            const nodeToPlayFromIndex = moveTree.getNodeIndex(nodeToPlayFrom);
            
            if (nodeToPlayFromIndex !== -1 && nodeToPlayFromIndex + 1 < moveTree.mainline.length) {
                const nextMainlineMove = moveTree.mainline[nodeToPlayFromIndex + 1];
                if (nextMainlineMove.move && 
                    nextMainlineMove.move.from === moveObj.from && 
                    nextMainlineMove.move.to === moveObj.to && 
                    nextMainlineMove.move.promotion === moveObj.promotion) {
                    existingNode = nextMainlineMove;
                }
            }
            
            // If not found in mainline, check for existing child in variations
            if (!existingNode) {
                existingNode = nodeToPlayFrom.children.find(child =>
                    child.move &&
                    child.move.from === moveObj.from &&
                    child.move.to === moveObj.to &&
                    child.move.promotion === moveObj.promotion
                );
            }

            if (existingNode) {
                nodeToPlayFrom = existingNode;
            } else {
                // Create and evaluate new node
                nodeToPlayFrom = this.createEngineLineNode(moveObj, chess, nodeToPlayFrom, moveTree, handleTreeNodeClick, queueMoveForEvaluation);
            }
        }

        // Navigate to final node
        handleTreeNodeClick(nodeToPlayFrom);
        // Note: showClassification will be handled by handleTreeNodeClick's updateAfterMove
        moveTree.updateCurrentMove(nodeToPlayFrom.id);
    }

    static createEngineLineNode(moveObj, chess, parentNode, moveTree, handleTreeNodeClick, queueMoveForEvaluation) {
        const newNode = moveTree.addMove(moveObj, parentNode.id);
        newNode.evaluationStatus = 'pending';

        // Update UI - Note: showClassification will be handled by handleTreeNodeClick's updateAfterMove
        moveTree.updateCurrentMove(newNode.id);

        // Queue for evaluation
        const fen = chess.fen();
        const prevFen = parentNode.fen;
        queueMoveForEvaluation(newNode, fen, prevFen);

        return newNode;
    }
}