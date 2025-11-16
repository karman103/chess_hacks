import { Chess } from '../../libs/chess.js';

export class GamePhase {
    constructor() {}

    static getPhases(moves) {
        if (!moves || moves.length === 0) {
            return [];
        }

        const phases = [];
        const chess = new Chess();
        
        // Track phase changes
        let currentPhase = 'opening';
        
        for (let i = 0; i < moves.length; i++) {
            // Make the move
            chess.move(moves[i]);
            
            // Determine phase at this position
            const phaseInfo = this._analyzePosition(chess, i + 1);
            const detectedPhase = phaseInfo.phase;
            
            // Check for phase transition
            if (detectedPhase !== currentPhase) {
                phases.push({
                    phase: detectedPhase,
                    startMove: i + 1,
                    moveNumber: Math.ceil((i + 1) / 2),
                    ply: i + 1,
                    reason: phaseInfo.reason,
                    criteria: phaseInfo.criteria
                });
                currentPhase = detectedPhase;
            }
        }
        
        // If no phases were detected, the entire game is in opening
        if (phases.length === 0) {
            phases.push({
                phase: 'opening',
                startMove: 1,
                moveNumber: 1,
                ply: 1,
                reason: 'Game start',
                criteria: { moveCount: 1 }
            });
        }
        
        return phases;
    }

    static _analyzePosition(chess, moveCount) {
        const board = chess.board();
        const fen = chess.fen();
        const moveNumber = chess.moveNumber();
        
        // Calculate various position metrics
        const metrics = this._calculateMetrics(board, chess, moveCount);
        
        // Determine phase based on multiple criteria
        if (this._isEndgame(metrics)) {
            return {
                phase: 'endgame',
                reason: this._getEndgameReason(metrics),
                criteria: metrics
            };
        } else if (this._isMiddlegame(metrics)) {
            return {
                phase: 'middlegame', 
                reason: this._getMiddlegameReason(metrics),
                criteria: metrics
            };
        } else {
            return {
                phase: 'opening',
                reason: this._getOpeningReason(metrics),
                criteria: metrics
            };
        }
    }

    static _calculateMetrics(board, chess, moveCount) {
        const metrics = {
            moveCount: moveCount,
            moveNumber: chess.moveNumber(),
            totalPieces: 0,
            developedPieces: { white: 0, black: 0 },
            materialValue: { white: 0, black: 0 },
            hasQueens: { white: false, black: false },
            hasCastled: { white: false, black: false },
            minorPiecesLeft: { white: 0, black: 0 },
            majorPiecesLeft: { white: 0, black: 0 }
        };

        // Piece values for material calculation
        const pieceValues = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
        
        // Starting positions for development tracking
        const startingPositions = {
            white: {
                knights: ['b1', 'g1'],
                bishops: ['c1', 'f1'],
                rooks: ['a1', 'h1'],
                queen: 'd1',
                king: 'e1'
            },
            black: {
                knights: ['b8', 'g8'], 
                bishops: ['c8', 'f8'],
                rooks: ['a8', 'h8'],
                queen: 'd8',
                king: 'e8'
            }
        };

        // Analyze each square on the board
        for (let rank = 0; rank < 8; rank++) {
            for (let file = 0; file < 8; file++) {
                const piece = board[rank][file];
                if (piece) {
                    metrics.totalPieces++;
                    const color = piece.color === 'w' ? 'white' : 'black';
                    const pieceType = piece.type.toLowerCase();
                    const square = String.fromCharCode(97 + file) + (8 - rank);
                    
                    // Add to material count
                    metrics.materialValue[color] += pieceValues[pieceType] || 0;
                    
                    // Count piece types
                    if (pieceType === 'q') {
                        metrics.hasQueens[color] = true;
                        metrics.majorPiecesLeft[color]++;
                    } else if (pieceType === 'r') {
                        metrics.majorPiecesLeft[color]++;
                    } else if (pieceType === 'n' || pieceType === 'b') {
                        metrics.minorPiecesLeft[color]++;
                    }
                    
                    // Check development
                    const startingPos = startingPositions[color];
                    if (startingPos) {
                        let isDeveloped = false;
                        
                        if (pieceType === 'n' && !startingPos.knights.includes(square)) {
                            isDeveloped = true;
                        } else if (pieceType === 'b' && !startingPos.bishops.includes(square)) {
                            isDeveloped = true;
                        } else if (pieceType === 'q' && square !== startingPos.queen) {
                            isDeveloped = true;
                        }
                        
                        if (isDeveloped) {
                            metrics.developedPieces[color]++;
                        }
                    }
                }
            }
        }

        // Check castling status
        const castlingRights = chess.getCastlingRights('w');
        const blackCastlingRights = chess.getCastlingRights('b');
        
        // If no castling rights remain, assume they've castled
        // (This is a simplification - could be improved by tracking king/rook moves)
        metrics.hasCastled.white = castlingRights.k === false && castlingRights.q === false;
        metrics.hasCastled.black = blackCastlingRights.k === false && blackCastlingRights.q === false;

        return metrics;
    }

    static _isEndgame(metrics) {
        // Endgame criteria (any of these conditions)
        const conditions = [
            // No queens on board
            !metrics.hasQueens.white && !metrics.hasQueens.black,
            
            // Very few pieces left (less than 16 total)
            metrics.totalPieces <= 12,
            
            // Low material count for both sides
            (metrics.materialValue.white + metrics.materialValue.black) <= 20,
            
            // Queens traded and few minor pieces
            !metrics.hasQueens.white && !metrics.hasQueens.black && 
            (metrics.minorPiecesLeft.white + metrics.minorPiecesLeft.black) <= 4,
            
            // Late in the game with limited material
            metrics.moveNumber >= 40 && metrics.totalPieces <= 16
        ];

        return conditions.some(condition => condition);
    }

    static _isMiddlegame(metrics) {
        // Middlegame criteria
        const conditions = [
            // Sufficient development and move count
            metrics.moveCount >= 10 && 
            (metrics.developedPieces.white >= 2 || metrics.developedPieces.black >= 2),
            
            // Castling has occurred for at least one side
            metrics.hasCastled.white || metrics.hasCastled.black,
            
            // Queens still on board but some development
            (metrics.hasQueens.white || metrics.hasQueens.black) && metrics.moveCount >= 8,
            
            // Sufficient pieces and material for middlegame complexity
            metrics.totalPieces >= 20 && metrics.moveCount >= 12
        ];

        return conditions.some(condition => condition) && !this._isEndgame(metrics);
    }

    static _getEndgameReason(metrics) {
        if (!metrics.hasQueens.white && !metrics.hasQueens.black) {
            return 'Queens traded off';
        } else if (metrics.totalPieces <= 12) {
            return `Few pieces remaining (${metrics.totalPieces})`;
        } else if ((metrics.materialValue.white + metrics.materialValue.black) <= 20) {
            return 'Low material count';
        } else {
            return 'Endgame characteristics detected';
        }
    }

    static _getMiddlegameReason(metrics) {
        if (metrics.hasCastled.white || metrics.hasCastled.black) {
            return 'Castling completed, pieces developed';
        } else if (metrics.developedPieces.white >= 2 || metrics.developedPieces.black >= 2) {
            return 'Sufficient piece development';
        } else if (metrics.moveCount >= 10) {
            return 'Opening phase complete';
        } else {
            return 'Middlegame characteristics detected';
        }
    }

    static _getOpeningReason(metrics) {
        if (metrics.moveCount <= 8) {
            return 'Early game development';
        } else if (metrics.developedPieces.white < 2 && metrics.developedPieces.black < 2) {
            return 'Limited piece development';
        } else {
            return 'Opening principles still apply';
        }
    }

    // Utility method to get current phase for a position
    static getCurrentPhase(chess) {
        const moveCount = chess.history().length;
        if (moveCount === 0) {
            return {
                phase: 'opening',
                reason: 'Game start',
                criteria: { moveCount: 0 }
            };
        }
        
        return this._analyzePosition(chess, moveCount);
    }

    // Get phase summary for the entire game
    static getGameSummary(moves) {
        const phases = this.getPhases(moves);
        
        const summary = {
            totalMoves: moves.length,
            totalPlies: moves.length,
            phases: phases,
            phaseDurations: {}
        };

        // Calculate phase durations
        for (let i = 0; i < phases.length; i++) {
            const phase = phases[i];
            const nextPhase = phases[i + 1];
            const endMove = nextPhase ? nextPhase.startMove - 1 : moves.length;
            const duration = endMove - phase.startMove + 1;
            
            if (!summary.phaseDurations[phase.phase]) {
                summary.phaseDurations[phase.phase] = 0;
            }
            summary.phaseDurations[phase.phase] += duration;
        }

        return summary;
    }
}