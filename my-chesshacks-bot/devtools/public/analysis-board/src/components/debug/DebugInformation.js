import { Chess } from "../../../libs/chess.js";

export class DebugInformation {
    constructor() {}

    /**
     * Updates the move probabilities display.
     * @param {Object} node - The current move node.
     * @param {Object} prevNode - The previous move node.
     * @param {Object} board - The chess board instance (optional, for hover interactions).
     * @param {boolean} showProbabilityArrows - Whether probability arrows are enabled.
     */
    static updateMoveProbabilities(node, prevNode, board, showProbabilityArrows = true) {
        const $probContainer = $(".debug-probabilities");
        $probContainer.empty();

        // Add section title
        const $title = $("<div>").addClass("section-title")
            .text("Move Probabilities");
        $probContainer.append($title);

        // Show placeholder when at the root node or have no probability data
        if (!node || node.id === 'root' || !node.probabilities || Object.keys(node.probabilities).length === 0) {
            const $placeholder = $("<div>").addClass("debug-info-placeholder")
                .text("Select a move to see probability distribution.");
            $probContainer.append($placeholder);
            return;
        }

        // Create content container
        const $content = $("<div>").addClass("debug-probability-content");
        
        // Get the best move UCI from the previous node's evaluation
        let bestMoveUci = null;
        if (prevNode?.evaluatedMove?.lines?.length > 0) {
            const bestLine = prevNode.evaluatedMove.lines.find(line => line.id === 1);
            if (bestLine) {
                bestMoveUci = bestLine.uciMove || (bestLine.pv && bestLine.pv.length > 0 ? bestLine.pv[0] : null);
            }
        }
        
        // Sort probabilities by value (descending)
        const sortedProbs = Object.entries(node.probabilities)
            .sort((a, b) => b[1] - a[1]);
        
        sortedProbs.forEach(([uciMove, prob]) => {
            const isBestMove = bestMoveUci === uciMove;
            const percentage = (prob * 100).toFixed(2);
            const percentageWidth = Math.min(prob * 100, 100);
            const $probLine = $("<div>")
                .addClass("debug-probability-line")
                .attr("data-uci-move", uciMove);
            
            // Convert UCI to SAN for display
            let sanMove = uciMove;
            try {
                const tempChess = new Chess(prevNode?.fen || node.fen);
                const from = uciMove.substring(0, 2);
                const to = uciMove.substring(2, 4);
                const promotion = uciMove.length > 4 ? uciMove.substring(4, 5) : undefined;
                const moveObj = tempChess.move({ from, to, promotion });
                if (moveObj) {
                    sanMove = moveObj.san;
                }
            } catch (e) {
                // If conversion fails, use UCI
            }
            
            // Add hover effect for arrows if board is available and arrows are enabled
            if (board && showProbabilityArrows) {
                $probLine.on("mouseenter", function() {
                    DebugInformation.highlightProbabilityArrow(board, node, prevNode, uciMove);
                });
                
                $probLine.on("mouseleave", function() {
                    DebugInformation.updateProbabilityArrows(board, node, prevNode);
                });
            }
            
            // Move name container with optional best move indicator
            const $moveContainer = $("<div>").css({
                "display": "flex",
                "align-items": "center",
                "gap": "6px",
                "min-width": "50px",
                "flex-shrink": "0"
            });
            
            const $moveSpan = $("<span>")
                .addClass("debug-move")
                .text(sanMove);
            $moveContainer.append($moveSpan);
            
            // Add green dot indicator if this is the best move
            if (isBestMove) {
                const $bestIndicator = $("<span>")
                    .css({
                        "width": "6px",
                        "height": "6px",
                        "border-radius": "50%",
                        "background-color": "rgb(34, 197, 94)",
                        "flex-shrink": "0",
                        "box-shadow": "0 0 4px rgba(34, 197, 94, 0.5)"
                    })
                    .attr("title", "Best move");
                $moveContainer.append($bestIndicator);
            }
            
            // Progress bar container
            const $progressContainer = $("<div>").addClass("debug-progress-container");
            const $progressBar = $("<div>")
                .addClass("debug-progress-bar")
                .css("width", `${percentageWidth}%`);
            $progressContainer.append($progressBar);
            
            // Percentage text
            const $percentageSpan = $("<span>")
                .addClass("debug-percentage")
                .text(`${percentage}%`);
            
            $probLine.append($moveContainer);
            $probLine.append($progressContainer);
            $probLine.append($percentageSpan);
            $content.append($probLine);
        });
        
        $probContainer.append($content);
    }

    /**
     * Updates the engine output/logs display.
     * @param {Object} node - The current move node.
     */
    static updateEngineOutput(node) {
        const $logsContainer = $(".debug-logs-section");
        $logsContainer.empty();

        // Create header container with title and copy button
        const $headerContainer = $("<div>").addClass("debug-header-container");
        
        // Add section title
        const $title = $("<div>").addClass("section-title")
            .text("Engine Output");
        $headerContainer.append($title);

        // Add copy button if we have logs
        if (node && node.id !== 'root' && node.logs) {
            const $copyButton = $("<button>")
                .addClass("debug-copy-button")
                .attr("title", "Copy to clipboard")
                .html(`
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512">
                        <path d="M384 336H192c-8.8 0-16-7.2-16-16V64c0-8.8 7.2-16 16-16l140.1 0L400 115.9V320c0 8.8-7.2 16-16 16zM192 384H384c35.3 0 64-28.7 64-64V115.9c0-12.7-5.1-24.9-14.1-33.9L366.1 14.1c-9-9-21.2-14.1-33.9-14.1H192c-35.3 0-64 28.7-64 64V320c0 35.3 28.7 64 64 64zM64 128c-35.3 0-64 28.7-64 64V448c0 35.3 28.7 64 64 64H256c35.3 0 64-28.7 64-64V416H272v32c0 8.8-7.2 16-16 16H64c-8.8 0-16-7.2-16-16V192c0-8.8 7.2-16 16-16H96V128H64z" fill="currentColor"/>
                    </svg>
                `)
                .on("click", function() {
                    navigator.clipboard.writeText(node.logs).catch(err => {
                        console.error('Failed to copy:', err);
                    });
                });
            $headerContainer.append($copyButton);
        }
        
        $logsContainer.append($headerContainer);

        // Show placeholder when at the root node or have no logs
        if (!node || node.id === 'root' || !node.logs) {
            const $placeholder = $("<div>").addClass("debug-info-placeholder")
                .text("Select a move to see engine output.");
            $logsContainer.append($placeholder);
            return;
        }

        // Add logs content
        const $logsContent = $("<pre>").addClass("debug-logs-content")
            .text(node.logs);
        $logsContainer.append($logsContent);
    }

    /**
     * Updates probability arrows on the chess board.
     * Shows the move probabilities for the current position.
     * @param {Object} board - The chess board instance.
     * @param {Object} node - The current move node.
     * @param {Object} prevNode - The previous move node (to get best move).
     * @param {string} highlightedMove - Optional UCI move to highlight (brighter/more opaque).
     */
    static updateProbabilityArrows(board, node, prevNode, highlightedMove = null) {
        // Clear any existing probability arrows
        board.clearProbabilityArrows();

        // Only show probability arrows if we have probability data
        if (!node || node.id === 'root' || !node.probabilities || Object.keys(node.probabilities).length === 0) {
            return;
        }

        // Get the best move UCI from the previous node's evaluation (same logic as best move arrow)
        let bestMoveUci = null;
        if (prevNode?.evaluatedMove?.lines?.length > 0) {
            const bestLine = prevNode.evaluatedMove.lines.find(line => line.id === 1);
            if (bestLine) {
                bestMoveUci = bestLine.uciMove || (bestLine.pv && bestLine.pv.length > 0 ? bestLine.pv[0] : null);
            }
        }

        // Sort probabilities by value (descending)
        const sortedProbs = Object.entries(node.probabilities)
            .sort((a, b) => b[1] - a[1]);
        
        // Show arrows for top moves (limit to top 10 to avoid clutter)
        const maxArrows = 10;
        const topMoves = sortedProbs.slice(0, maxArrows);

        topMoves.forEach(([uciMove, prob]) => {
            try {
                const from = uciMove.substring(0, 2);
                const to = uciMove.substring(2, 4);
                
                const fromIdx = board.algebraicToIndex(from, board.flipped);
                const toIdx = board.algebraicToIndex(to, board.flipped);
                
                const fromSquare = board.getSquare(fromIdx);
                const toSquare = board.getSquare(toIdx);
                
                if (fromSquare && toSquare) {
                    const isHighlighted = highlightedMove === uciMove;
                    const isBestMove = bestMoveUci === uciMove;
                    
                    // Use blue color with opacity based on probability
                    // Higher probability = more opaque (closer to 1.0)
                    // Scale opacity: minimum 0.4, maximum 0.9
                    let opacity = Math.max(0.4, Math.min(0.9, prob * 2.5));
                    let color = 'rgb(100, 149, 237)'; // Cornflower blue
                    
                    // If this move is the best move, make it GREEN!
                    if (isBestMove) {
                        color = 'rgb(34, 197, 94)'; // Green (emerald-500)
                        opacity = Math.max(0.7, opacity); // Ensure it's visible
                    }
                    
                    // If this move is highlighted (hovered), make it brighter and more opaque
                    if (isHighlighted) {
                        opacity = 1.0; // Full opacity on hover
                        if (isBestMove) {
                            color = 'rgb(74, 222, 128)'; // Lighter green (emerald-400) on hover
                        } else {
                            color = 'rgb(135, 206, 250)'; // Light sky blue (brighter) on hover
                        }
                    }
                    
                    board.addArrow(fromSquare, toSquare, {
                        color: color,
                        opacity: opacity,
                        isProbability: true
                    });
                }
            } catch (e) {
                console.error('Failed to draw probability arrow for move:', uciMove, e);
            }
        });
    }

    /**
     * Highlights a specific probability arrow on hover.
     * @param {Object} board - The chess board instance.
     * @param {Object} node - The current move node.
     * @param {Object} prevNode - The previous move node (to get best move).
     * @param {string} uciMove - The UCI move to highlight.
     */
    static highlightProbabilityArrow(board, node, prevNode, uciMove) {
        DebugInformation.updateProbabilityArrows(board, node, prevNode, uciMove);
    }

    /**
     * Updates all debug information displays.
     * @param {Object} node - The current move node.
     * @param {Object} prevNode - The previous move node.
     * @param {Object} board - The chess board instance (optional, for probability arrows and hover effects).
     * @param {boolean} showProbabilityArrows - Whether to show probability arrows on the board.
     */
    static updateDebugInfo(node, prevNode, board, showProbabilityArrows = true) {
        DebugInformation.updateMoveProbabilities(node, prevNode, board, showProbabilityArrows);
        DebugInformation.updateEngineOutput(node);
        
        // Show probability arrows on the board if available and enabled
        if (board && showProbabilityArrows) {
            DebugInformation.updateProbabilityArrows(board, node, prevNode);
        } else if (board && !showProbabilityArrows) {
            // Clear probability arrows if the setting is disabled
            board.clearProbabilityArrows();
        }
    }
}

