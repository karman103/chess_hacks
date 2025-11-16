import { Chess } from '../../libs/chess.js';
import { EvaluationBar } from '../components/board/EvaluationBar.js';
import { chessOpeningTree } from '../data/openings.js';
import { MoveAnnotator } from './MoveAnnotator.js';

export const ClasifCss = {
	MOVE_THEORY: "theory-move",
    MOVE_BRILLIANT: "brilliant-move",
    MOVE_GREAT: "great-move",
    MOVE_PERFECT: "perfect-move",
    MOVE_EXCELLENT: "excellent-move",
    MOVE_GOOD: "good-move",
    MOVE_INACCURACY: "inaccuracy-move",
    MOVE_MISTAKE: "mistake-move",
    MOVE_BLUNDER: "blunder-move",
    MOVE_FORCED: "forced-move",
    MOVE_MISS: "miss-move",
};

export const Classification = {
    BRILLIANT: {
        type: "brilliant",
        src: "/classifications/brilliant.svg",
        class: ClasifCss.MOVE_BRILLIANT,
        accuracy: 1,
        comment: "is a brilliant move!",
        color: "#14e6e6"
    },
    GREAT: {
        type: "great",
        src: "/classifications/great.svg",
        class: ClasifCss.MOVE_GREAT,
        accuracy: 1,
        comment: "is a great move!",
        color: "#38a5ff"
    },
    PERFECT: {
        type: "perfect",
        src: "/classifications/perfect.svg",
        class: ClasifCss.MOVE_PERFECT,
        accuracy: 1,
        comment: "is perfect.",
        color: "#7bcc18"
    },
    EXCELLENT: {
        type: "excellent",
        src: "/classifications/excellent.svg",
        class: ClasifCss.MOVE_EXCELLENT,
        accuracy: 0.9,
        comment: "is an excellent move.",
        color: "#7bcc18"
    },
    GOOD: {
        type: "good",
        src: "/classifications/good.svg",
        class: ClasifCss.MOVE_GOOD,
        accuracy: 0.7,
        comment: "is a good move.",
        color: "#088a28"
    },
    THEORY: {
        type: "theory",
        src: "/classifications/theory.svg",
        class: ClasifCss.MOVE_THEORY,
        accuracy: 1,
        comment: "is theory.",
        color: "#e09d0d"
    },
    INACCURACY: {
        type: "inaccuracy",
        src: "/classifications/inaccuracy.svg",
        class: ClasifCss.MOVE_INACCURACY,
        accuracy: 0.4,
        comment: "is an inaccuracy.",
        color: "#ddd015"
    },
    MISTAKE: {
        type: "mistake",
        src: "/classifications/mistake.svg",
        class: ClasifCss.MOVE_MISTAKE,
        accuracy: 0.2,
        comment: "is a mistake.",
        color: "#e5820d"
    },
    BLUNDER: {
        type: "blunder",
        src: "/classifications/blunder.svg",
        class: ClasifCss.MOVE_BLUNDER,
        accuracy: 0,
        comment: "was a blunder!",
        color: "#d44242"
    },
    FORCED: {
        type: "forced",
        src: "/classifications/forced.svg",
        class: ClasifCss.MOVE_FORCED,
        accuracy: 1,
        comment: "was forced.",
        color: "#088a28"
    },
    MISS: {
        type: "miss",
        src: "/classifications/miss.svg",
        class: ClasifCss.MOVE_MISS,
        accuracy: 0,
        comment: "was a miss.",
        color: "#d44242"
    }
};

export const CommentType = {
    NONE: "none",
    MISS: "miss", // Missed mate
    STALEMATE: "stalemate",
    WON: "won", // Won the game
    FORCED: "forced",
    GOT_MATED: "got_mated",
    WILL_MATE: "will_mate",
    STILL_WINNING: "still_winning"
}

export class MoveClassifier {
    // Classification types with no special rules
    static centipawnClassifications = [
        Classification.PERFECT,
        Classification.EXCELLENT,
        Classification.GOOD,
        Classification.INACCURACY,
        Classification.MISTAKE,
        Classification.BLUNDER
    ];

    static pieceValues = {
        'p': 1,
        'n': 3,
        'b': 3,
        'r': 5,
        'q': 9,
        'k': 1000
    };

    // Get the maximum evaluation loss for a classification to be applied
    // Evaluation loss threshold for excellent in a previously equal position is 30
    // These numbers are from the Game Report github repo by Wintrcat
    static evalLossThresholds = {
        [Classification.PERFECT.type]: (prevEval) => Math.max(0.0001 * Math.pow(Math.abs(prevEval), 2) + 0.0236 * Math.abs(prevEval) - 3.7143, 0),
        [Classification.EXCELLENT.type]: (prevEval) => Math.max(0.0002 * Math.pow(Math.abs(prevEval), 2) + 0.1231 * Math.abs(prevEval) + 27.5455, 0),
        [Classification.GOOD.type]: (prevEval) => Math.max(0.0002 * Math.pow(Math.abs(prevEval), 2) + 0.2643 * Math.abs(prevEval) + 60.5455, 0),
        [Classification.INACCURACY.type]: (prevEval) => Math.max(0.0002 * Math.pow(Math.abs(prevEval), 2) + 0.3624 * Math.abs(prevEval) + 108.0909, 0),
        [Classification.MISTAKE.type]: (prevEval) => Math.max(0.0003 * Math.pow(Math.abs(prevEval), 2) + 0.4027 * Math.abs(prevEval) + 225.8182, 0),
        [Classification.BLUNDER.type]: () => Infinity
    };

    static getAttackers(chess, square, color) {
        return chess.moves({ verbose: true }).filter(m => m.to === square && m.color === color);
    }

    static getDefenders(chess, square, targetColor) {
        // Create a copy with swapped piece color to find defenders
        const chessDef = new Chess(chess.fen());
        const originalPiece = chessDef.get(square);
        chessDef.remove(square);
        chessDef.put({ type: originalPiece.type, color: targetColor === 'w' ? 'b' : 'w' }, square);
        
        // Set turn to target color to generate defending moves
        const fenParts = chessDef.fen().split(' ');
        fenParts[1] = targetColor;
        const defenderChess = new Chess(fenParts.join(' '));
        
        return defenderChess.moves({ verbose: true }).filter(m => m.to === square && m.color === targetColor);
    }

    static isKingAdjacent(kingSquare, targetSquare) {
        if (!kingSquare) return false;
        
        const fileDiff = Math.abs(kingSquare.charCodeAt(0) - targetSquare.charCodeAt(0));
        const rankDiff = Math.abs(parseInt(kingSquare[1]) - parseInt(targetSquare[1]));
        
        return fileDiff <= 1 && rankDiff <= 1 && (fileDiff > 0 || rankDiff > 0);
    }

    static findKingSquare(chess, color) {
        for (const row of chess.board()) {
            for (const pieceObj of row) {
                if (pieceObj?.type === 'k' && pieceObj.color === color) {
                    return pieceObj.square;
                }
            }
        }
        return null;
    }

    static calculateMaterialExchange(targetValue, attackerValues, defenderValues) {
        let netMaterial = -targetValue;
        let attackerIndex = 0;
        let defenderIndex = 0;
        let isEnemyTurn = true;

        while (attackerIndex < attackerValues.length || defenderIndex < defenderValues.length) {
            if (isEnemyTurn) {
                if (attackerIndex >= attackerValues.length) break;
                netMaterial -= attackerValues[attackerIndex++];
            } else {
                if (defenderIndex >= defenderValues.length) break;
                netMaterial += defenderValues[defenderIndex++];
            }
            isEnemyTurn = !isEnemyTurn;
        }

        return netMaterial;
    }

    /**
     * Checks if a piece is hanging (can be captured without adequate compensation)
     * @param {string} fen - FEN string of the position
     * @param {string} square - Algebraic notation of the square to check
     * @returns {boolean} - True if the piece is hanging
     */
    static isPieceHanging(fen, square) {
        const chess = new Chess(fen);
        const piece = chess.get(square);
        if (!piece) return false;

        const { type: targetPiece, color: targetColor } = piece;
        const enemyColor = targetColor === 'w' ? 'b' : 'w';

        // Get attackers and defenders
        const enemyAttackers = this.getAttackers(chess, square, enemyColor);
        if (enemyAttackers.length === 0) return false;

        const friendlyDefenders = this.getDefenders(chess, square, targetColor);

        // Add king as potential attacker if adjacent
        const kingSquare = this.findKingSquare(chess, enemyColor);
        if (this.isKingAdjacent(kingSquare, square)) {
            enemyAttackers.push({ piece: 'k', color: enemyColor });
        }

        // Quick check: more attackers than defenders
        if (enemyAttackers.length > friendlyDefenders.length) return true;

        // Calculate material exchange
        const targetValue = this.pieceValues[targetPiece];
        const attackerValues = enemyAttackers.map(m => this.pieceValues[m.piece]).sort((a, b) => a - b);
        const defenderValues = friendlyDefenders.map(m => this.pieceValues[m.piece]).sort((a, b) => a - b);

        const netMaterial = this.calculateMaterialExchange(targetValue, attackerValues, defenderValues);
        
        return netMaterial >= 0;
    }

    /**
     * Finds pieces that have been sacrificed in the current position
     * @param {Chess} currentBoard - The current board position
     * @param {boolean} isBlack - Whether the current player is black
     * @param {Object} lastPiece - The piece that was on the destination square in the previous position (captured piece)
     * @param {string} fen - FEN string of the current position
     * @returns {Array} - Array of sacrificed pieces
     */
    static findSacrificedPieces(currentBoard, isBlack, lastPiece, fen) {
        const sacrificedPieces = [];
        
        for (let row of currentBoard.board()) {
            for (let piece of row) {
                // Skip empty squares
                if (!piece) continue;
                
                // Skip opponent pieces
                if (piece.color != (isBlack ? 'w' : 'b')) continue;
                
                // Skip kings and pawns
                if (piece.type == "k" || piece.type == "p") continue;
                
                // Skip if the captured piece was more valuable
                if (lastPiece && this.pieceValues[lastPiece.type] >= this.pieceValues[piece.type]) {
                    continue;
                }
                
                // Check if piece is hanging
                if (this.isPieceHanging(fen, piece.square)) {
                    sacrificedPieces.push(piece);
                }
            }
        }
        
        return sacrificedPieces;
    }
    
    /**
     * Checks if a move is damage control rather than a genuine brilliant sacrifice
     * @param {Object} movedPiece - The piece that was moved
     * @param {Array} sacrificedPieces - Array of pieces that are now hanging
     * @param {string} uciMove - The UCI move string
     * @param {string} previousFen - FEN string of the position before the move
     * @returns {boolean} - True if this is damage control, not a genuine sacrifice
     */
    static isDamageControlMove(movedPiece, sacrificedPieces, uciMove, previousFen) {
        if (!movedPiece || !sacrificedPieces.length) return false;
        
        const movedPieceValue = this.pieceValues[movedPiece.type];
        const maxSacrificedValue = Math.max(...sacrificedPieces.map(p => this.pieceValues[p.type]));
        const totalSacrificedValue = sacrificedPieces.reduce((sum, p) => sum + this.pieceValues[p.type], 0);
        
        // If the moved piece is more valuable than what's being "sacrificed"
        if (movedPieceValue > maxSacrificedValue) {
            const fromSquare = uciMove.slice(0, 2);
            
            // Check if the moved piece was under attack in the previous position
            if (this.isPieceHanging(previousFen, fromSquare)) {
                return true; // Damage control: saved more valuable piece, left less valuable one
            }
            
            // Additional check: if multiple pieces are being "sacrificed" and their combined value
            // is less than the moved piece, this is likely damage control
            // if (sacrificedPieces.length > 1 && totalSacrificedValue < movedPieceValue * 0.8) {
            //     return true;
            // }
        }
        
        // Check for pattern where a high-value piece moves away from a square that's under attack
        // and leaves behind lower-value pieces that were previously protected
        if (movedPieceValue >= 6) { // Rook or higher value
            const chess = new Chess(previousFen);
            const fromSquare = uciMove.slice(0, 2);
            
            // // Check if any of the sacrificed pieces were on squares that the moved piece could defend
            // for (const sacrificedPiece of sacrificedPieces) {
            //     if (this.couldPieceDefend(chess, fromSquare, sacrificedPiece.square, movedPiece.type)) {
            //         // The moved piece was potentially defending the sacrificed piece
            //         // Moving it away and leaving the other piece hanging suggests damage control
            //         return true;
            //     }
            // }
        }
        
        return false;
    }
    
    /**
     * Checks if a piece could potentially defend another square
     * @param {Chess} chess - Chess instance
     * @param {string} defenderSquare - Square of the potential defender
     * @param {string} targetSquare - Square that might be defended
     * @param {string} pieceType - Type of the defending piece
     * @returns {boolean} - True if the piece could defend the target square
     */
    static couldPieceDefend(chess, defenderSquare, targetSquare, pieceType) {
        // Simple check based on piece movement patterns
        const defenderFile = defenderSquare.charCodeAt(0);
        const defenderRank = parseInt(defenderSquare[1]);
        const targetFile = targetSquare.charCodeAt(0);
        const targetRank = parseInt(targetSquare[1]);
        
        const fileDiff = Math.abs(defenderFile - targetFile);
        const rankDiff = Math.abs(defenderRank - targetRank);
        
        // Fast, but not super accurate. Still helps a lot.
        switch (pieceType) {
            case 'q': // Queen can defend any square in line
                return fileDiff === 0 || rankDiff === 0 || fileDiff === rankDiff;
            case 'r': // Rook can defend same file or rank
                return fileDiff === 0 || rankDiff === 0;
            case 'b': // Bishop can defend diagonal
                return fileDiff === rankDiff;
            case 'n': // Knight has specific L-shaped moves
                return (fileDiff === 2 && rankDiff === 1) || (fileDiff === 1 && rankDiff === 2);
            case 'k': // King can defend adjacent squares
                return fileDiff <= 1 && rankDiff <= 1;
            case 'p': // Pawn can defend diagonally
                return fileDiff === 1 && rankDiff === 1;
            default:
                return false;
        }
    }

    /**
     * Checks if any sacrificed piece can be safely captured by the opponent
     * @param {string} fen - FEN string of the current position
     * @param {Array} sacrificedPieces - Array of pieces that are considered sacrificed
     * @returns {Array} - Array of pieces that can be safely captured
     */
    static isAnySacrificeSafeToCaptureByOpponent(fen, sacrificedPieces) {
        const captureTestBoard = new Chess(fen);

        const trueSacrificedPieces = [];
        for (let piece of sacrificedPieces) {
            const attackers = captureTestBoard.moves({ verbose: true })
                .filter(m => m.to === piece.square && m.color !== piece.color);
                
            if (this.canAttackerSafelyCapture(captureTestBoard, attackers, piece, sacrificedPieces)) {
                trueSacrificedPieces.push(piece);
            }
        }
        
        return trueSacrificedPieces;
    }
    
    /**
     * Checks if any attacker can safely capture the sacrificed piece
     * @param {Chess} captureTestBoard - Chess instance to test captures
     * @param {Array} attackers - Array of possible attacking moves
     * @param {Object} piece - The piece being attacked
     * @param {Array} sacrificedPieces - All sacrificed pieces
     * @returns {boolean} - True if piece can be safely captured
     */
    static canAttackerSafelyCapture(captureTestBoard, attackers, piece, sacrificedPieces) {
        const promotions = ['q', 'r', 'b', 'n'];
        
        for (let attacker of attackers) {
            for (let promotion of promotions) {
                try {
                    captureTestBoard.move({
                        from: attacker.from,
                        to: piece.square,
                        promotion: promotion
                    });
                    
                    // Check if capturing piece would be pinned/vulnerable
                    const attackerPinned = this.isAttackerPinned(captureTestBoard, sacrificedPieces);
                    
                    // For high-value pieces (rook+), any safe capture is bad
                    if (this.pieceValues[piece.type] >= 2) {
                        if (!attackerPinned) {
                            return true;
                        }
                    } 

                    // For lower value pieces, check also that there's no immediate mate
                    else if (!attackerPinned && !captureTestBoard.moves().some(move => move.endsWith("#"))) {
                        return true;
                    }
                    
                    captureTestBoard.undo();
                } catch {}
            }
        }
        
        return false;
    }
    
    /**
     * Checks if the attacker that just captured would be vulnerable
     * @param {Chess} captureTestBoard - Chess instance after capture
     * @param {Array} sacrificedPieces - All sacrificed pieces
     * @returns {boolean} - True if attacking piece would be pinned/vulnerable
     */
    static isAttackerPinned(captureTestBoard, sacrificedPieces) {
        const maxSacrificeValue = Math.max(...sacrificedPieces.map(sack => this.pieceValues[sack.type]));
        
        for (let row of captureTestBoard.board()) {
            for (let enemyPiece of row) {
                // Skip empty squares, friendly pieces, kings and pawns
                if (!enemyPiece) continue;
                if (enemyPiece.color == captureTestBoard.turn()) continue;
                if (enemyPiece.type == "k" || enemyPiece.type == "p") continue;
                
                // Check if piece is hanging and worth more than the sacrifice
                if (this.isPieceHanging(captureTestBoard.fen(), enemyPiece.square) && 
                    this.pieceValues[enemyPiece.type] >= maxSacrificeValue) {
                    return true;
                }
            }
        }
        
        return false;
    }

    /**
     * Checks if the move sequence matches an opening in the opening database
     * @param {Array} moves - Array of moves in UCI format
     * @returns {boolean} - True if sequence matches known opening theory
     */
    static isInOpeningTheory(moves) {
        if (!moves || moves.length === 0) return false;
        
        let currentNode = chessOpeningTree;

        for (const move of moves) {
            if (currentNode[move]) {
                currentNode = currentNode[move];
            } else {
                return false;
            }
        }

        return true;
    }

    /**
     * Evaluates a move to determine its classification
     * @param {Object} move - Current move data
     * @param {Object} previous - Previous move data
     * @param {Array} moves - Array of all moves (strings) played up to (and including) the current move
     * @returns {Object} - Classification object for the move
     */
    static classifyMove(move, previous, moves) {
        // Prepare a default classification to return
        let classification = Classification.GOOD;
        
        // Get the best and second best lines
        const bestLine = move.lines.find(line => line.id === 1);
        const secondLine = move.lines.find(line => line.id === 2);
        const prevBestLine = previous.lines.find(line => line.id === 1);
        const prevSecondLine = previous.lines.find(line => line.id === 2);
        const isBlack = move.fen.includes(" b ");

        // For the fancy computer annotations
        move.commentData = {}
        move.commentType = CommentType.NONE;

        // If there's no first line, it's mate or draw
        if (!bestLine) {
            const board = new Chess(move.fen);
            if (board.isCheckmate()) {
                classification = Classification.PERFECT;
                move.commentType = CommentType.WON;
                move.graph = isBlack ? 0 : 100;
                move.win = 100;
            } else {
                move.graph = 50;
                move.win = 0; // You can't win/lose if it's a draw
                if (board.isStalemate()) {
                    classification = Classification.MISS;
                    move.commentType = CommentType.STALEMATE;
                } else {
                    // If it's a draw by repetition, and we're not in check and were winning, it's a blunder
                    if (board.isThreefoldRepetition() && !board.inCheck() && absEval > 0) {
                        classification = Classification.BLUNDER;
                    } else {
                        classification = Classification.PERFECT;
                    }
                }
            }
            move.classification = classification;
            
            return classification;
        }

        // Calculate win% for later on
        const win = 100 - (50 + 50 * (2 / (1 + Math.exp(-0.00368208 * (bestLine.score * (isBlack ? -1 : 1)))) - 1));
        move.win = win;

        // Graph Evaluation (lying to the user)
        move.graph = 100 - EvaluationBar.scoreToPercentage(bestLine.score);
        
        // Check if the move is part of opening theory
        if (previous.classification == Classification.THEORY && moves.length > 0 && this.isInOpeningTheory(moves)) {
            classification = Classification.THEORY;
            move.classification = classification;
            move.centipawnLoss = 0;
            return classification;
        }

        // Since mate has a different "score" thing
        if (bestLine.type == "mate") {
            move.graph = (bestLine.score > 0) ? 0 : 100;
        }
        
        // No second computer line means no other line exists (the move is forced)
        if (!prevSecondLine) {
            classification = Classification.FORCED;
            move.classification = classification;
            move.commentType = CommentType.FORCED;
            move.centipawnLoss = 0;
            return classification;
        }

        // If the move has no score, it's probably because the opening book didn't work
        if (move.score === null) {
            classification = Classification.GOOD;
            move.classification = classification;
            move.centipawnLoss = 0;
            return classification;
        }

        // Helper function to calculate the evaluation loss difference
        const diff = (prevScore, currentScore) => isBlack ? prevScore - currentScore : currentScore - prevScore;

        // Calculate the evaluation lost as a result of this move compared to the top computer moves
        const matchingTopLine = move.lines.find(line => line.uciMove === move.uciMove);
        const lastLineEvalLoss = matchingTopLine ? diff(prevBestLine.score, matchingTopLine.score) : Infinity;
        const evalLoss = Math.min(diff(prevBestLine.score, bestLine.score), lastLineEvalLoss);
        
        // Used for esimated Elo calculations
        move.centipawnLoss = evalLoss;

        // We'll use absolute eval and previous absolute eval to resolve unclear mate situations
        const noMate = prevBestLine.type == "cp" && bestLine.type == "cp";
        const absEval = bestLine.score * (isBlack ? 1 : -1);
        const prevAbsEval = prevBestLine.score * (isBlack ? 1 : -1);

        // If the move matches the top computer move already we can skip the rest
        if (move.uciMove === prevBestLine.uciMove) {
            classification = Classification.PERFECT;
        } else {
            if (noMate) {
                // Standard move evaluation
                classification = this.centipawnClassifications.find(classif => evalLoss <= this.evalLossThresholds[classif.type](0)) || classification;
            }

            // If no mate last move but you blundered a mate
            else if (prevBestLine.type == "cp" && bestLine.type == "mate") {
                move.commentData = { mateIn: absEval }
                if (absEval > 0) {
                    // Comment: Nothing you can do about it.
                    classification = Classification.PERFECT;
                    move.commentType = CommentType.GOT_MATED;
                } else if (absEval >= -2) {
                    // Comment: Missing mate in 2 is a blunder.
                    move.commentType = CommentType.GOT_MATED;
                    classification = Classification.BLUNDER;
                } else if (absEval >= -5) {
                    move.commentType = CommentType.GOT_MATED;
                    // Comment: Missing mate in 5 is a mistake.
                    classification = Classification.MISTAKE;
                } else {
                    move.commentType = CommentType.GOT_MATED;
                    // Comment: Missing mate in x is a inaccuracy. (did opponent catch it)
                    classification = Classification.INACCURACY;
                }
            }

            // If mate last move and there is no longer a mate
            else if (prevBestLine.type == "mate" && bestLine.type == "cp") {
                move.commentType = CommentType.MISS;
                if (prevAbsEval < 0 && absEval < 0) {
                    classification = Classification.PERFECT;
                } else if (absEval >= 400) {
                    classification = Classification.GOOD;
                } else if (absEval >= 200) {
                    classification = Classification.MISTAKE;
                } else {
                    classification = Classification.MISS;
                }
            }

            // If mate last move and forced mate still exists
            else if (prevBestLine.type == "mate" && bestLine.type == "mate") {
                move.commentData = { mateIn: absEval, prevMateIn: prevAbsEval }
                move.commentType = CommentType.WILL_MATE;
                if (prevAbsEval > 0) {
                    if (absEval <= -4) {
                        // There was a faster way to mate
                        classification = Classification.MISTAKE;
                    } else if (absEval < 0) {
                        classification = Classification.BLUNDER;
                    } else if (absEval < prevAbsEval) {
                        classification = Classification.PERFECT;
                    } else if (absEval <= prevAbsEval + 2) {
                        classification = Classification.EXCELLENT;
                    } else {
                        classification = Classification.GOOD;
                    }
                } else {
                    if (absEval == prevAbsEval) {
                        classification = Classification.PERFECT;
                    } else {
                        classification = Classification.GOOD;
                    }
                }
            }
        }

        // Check if blunder is due to throwing away a piece
        if (classification === Classification.BLUNDER) {
            const destinationSquare = move.uciMove.slice(2, 4);
            
            // Check if the piece that just moved is now hanging
            if (this.isPieceHanging(move.fen, destinationSquare)) {
                const board = new Chess(move.fen);
                const piece = board.get(destinationSquare);
                
                if (piece) {
                    // Record that this blunder threw away a piece
                    move.commentData.thrownAwayPiece = {
                        type: piece.type,
                        square: destinationSquare,
                        value: this.pieceValues[piece.type]
                    };
                }
            }
        }

        // The second 'great' move check
        if (prevSecondLine && classification === Classification.PERFECT && noMate) {
            // Was this move the only good move compared to the next two options?
            const evalDiff = Math.abs(prevBestLine.score - prevSecondLine.score);

            // If there was a piece hanging, the only "good" move is to capture it, so not exactly a hard to find move
            const wasPieceHanging = this.isPieceHanging(previous.fen, move.uciMove.slice(2, 4));

            if (evalDiff > 130 && !wasPieceHanging) {
                classification = Classification.GREAT;
            }
        }

        if (!secondLine) {
            move.classification = classification;
            return classification;
        }

        const secondAbsEval = secondLine.score * (isBlack ? 1 : -1);
        if (classification === Classification.PERFECT || classification === Classification.GREAT || classification === Classification.EXCELLENT) {
            // If the position is already overwhelmingly winning (e.g. mate scenarios or huge centipawn advantage)
            const winningAnyways = (secondAbsEval >= 900 && bestLine.type == "cp") || 
                                   (bestLine.type == "mate" && secondLine.type == "mate");
            const lastBoard = new Chess(previous.fen);

            // Skip brilliant move detection if evaluation is negative or position is winning anyway
            if (absEval < -50 || winningAnyways || lastBoard.inCheck()) {
                // Continue to next section - great move check
            } else {
                const currentBoard = new Chess(move.fen);

                // Get the piece that was captured (if any)
                const lastPiece = lastBoard.get(move.uciMove.slice(2, 4));
                
                // Find sacrificed pieces - friendly pieces that are now hanging
                let sacrificedPieces = this.findSacrificedPieces(currentBoard, isBlack, lastPiece, move.fen);

                // If we found sacrificed pieces, mark as brilliant and check if sacrifice is sound
                if (sacrificedPieces.length > 0) {
                    classification = Classification.BRILLIANT;
                    
                    // Check if this is damage control rather than a genuine sacrifice
                    const movedPiece = lastBoard.get(move.uciMove.slice(0, 2));
                    if (movedPiece && this.isDamageControlMove(movedPiece, sacrificedPieces, move.uciMove, previous.fen)) {
                        // This is damage control, not a brilliant sacrifice
                        classification = Classification.PERFECT;
                        sacrificedPieces = [];

                    } else {
                        // Check if any sacrificed piece can be safely captured
                        const piecesViablyCapturable = this.isAnySacrificeSafeToCaptureByOpponent(move.fen, sacrificedPieces);

                        move.commentData = { piecesViablyCapturable };

                        // If no sacrificed piece can be safely captured, revert to PERFECT
                        if (piecesViablyCapturable.length === 0) {
                            classification = Classification.PERFECT;
                            sacrificedPieces = sacrificedPieces.filter(piece => piece.square !== move.uciMove.slice(2, 4));
                        }
                    }
                }
            }
        }

        // Do not allow blunder if the game is still completely winning or losing
        const prevWin = 100 - (50 + 50 * (2 / (1 + Math.exp(-0.00368208 * (prevBestLine.score * (!isBlack ? -1 : 1)))) - 1));
        const prevOppositeWin = 100 - prevWin;
        if (classification == Classification.BLUNDER && (prevOppositeWin > 80 || prevOppositeWin < 20) && (win > 80 || win < 20)) {
            classification = Classification.INACCURACY;
            move.commentType = CommentType.STILL_WINNING;
        }

        // Store the classification on the move object for reference
        move.classification = classification;
        
        
        // Return the classification object
        return classification;
    }
} 