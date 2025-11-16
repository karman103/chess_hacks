import { Chess } from "../../libs/chess.js";
import { GamePhase } from "../classification/GamePhase.js";
import { MoveAnnotator } from "../classification/MoveAnnotator.js";
import { MoveClassifier, Classification } from "../classification/MoveClassifier.js";
import { Engine } from "./Engine.js";


// import { Engine } from './Engine.js';
// import { Evaluation } from '../classification/Evaluation.js';
// import { Classification } from "../classification/Classifications.js";

/**
 * Manages a queue of moves to be evaluated and handles background processing
 */
export class MoveEvaluator {
    constructor() {}

    static startPositionEvaluation = {
        fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        lines: [
            {
                id: 1,
                depth: 99,
                pv: ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "d7d5", "b1c3", "f8e7", "c1f4", "e8g8"],
                score: 19,
                type: "cp"
            },
            {
                id: 2,
                depth: 99,
                pv: ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1", "f6e4", "f1e1", "e4d6"],
                score: 18,
                type: "cp"
            },
            {
                id: 3,
                depth: 99,
                pv: ["c2c4", "g8f6", "g1f3", "e7e6", "d2d4", "d7d5", "b1c3", "f8e7", "c1f4", "e8g8"],
                score: 18,
                type: "cp"
            }
        ],
        engine: "Precomputed"
    }

    /**
     * Attempts to get a cloud evaluation from Lichess
     * @param {string} fen - FEN string to evaluate
     * @returns {Promise<Array|undefined>} - Array of evaluation lines or undefined if unavailable
     */
    static async tryCloudEvaluation(fen) {
        return false;
        // Send request with a short timeout to prevent long waits when rate limited
        return $.ajax({
            url: "https://lichess.org/api/cloud-eval",
            data: { fen, multiPv: 3 },
            method: "GET",
            dataType: "json",
            timeout: 2000, // Short timeout to avoid waiting too long
        }).then(({ depth, pvs }) => {
                const cloudUCIFixes = {
                    e8h8: "e8g8",
                    e1h1: "e1g1",
                    e8a8: "e8c8",
                    e1a1: "e1c1",
                };
                return pvs.map((pv, idx) => {
                    const moves = pv.moves.split(" ");
                    for (let i = 0; i < moves.length; i++) {
                        moves[i] = cloudUCIFixes[moves[i]] || moves[i];
                    }

                    const uciMove = moves[0];
                    const type = pv.cp === undefined ? "mate" : "cp";
                    const score = pv.cp === undefined ? pv.mate : pv.cp;
                    return { id: idx + 1, depth, uciMove, score, type, pv: moves };
                });
            })
            .catch(() => undefined);
    }

    /**
     * Processes a batch of moves for evaluation
     * @param {Chess} game - Chess instance 
     * @param {Array} history - Array of moves
     * @param {Function} progressCallback - Callback function for progress updates
     * @returns {Promise<Array>} - Array of evaluated moves
     */
    static async batchEvaluateMoves(game, history, progressCallback = null, settings = {}) {
        // Convert all moves to fen post-move, the move, and the index (to sort later)
        const queue = history.map((move, i) => {
            const moveObj = game.move(move);
            const fen = game.fen();
            const uciMove = moveObj.from + moveObj.to;

            return { move, fen, i, uciMove };
        });

        const depth = settings.engineDepth || 16;
        const engineType = settings.engineType || 'stockfish-17-lite';
        const maxWorkers = navigator.hardwareConcurrency || 8; // Use hardware concurrency when available
        const moves = new Array(history.length);
        
        // Create a pool of workers upfront and reuse them
        const workerPool = Array.from({ length: Math.min(maxWorkers, queue.length) }, () => new Engine({ engineType }));
        
        let completedMoves = 0;

        // Process moves in batches using the worker pool
        return new Promise((resolve) => {
            // Process a batch of positions with available workers
            async function processBatch() {
                const availableWorkers = workerPool.filter(worker => !worker.busy);
                
                // Process moves with available workers
                const batch = availableWorkers.map(worker => {
                    if (queue.length === 0) return null;
                    
                    const move = queue.shift();
                    worker.busy = true;
                    
                    return { worker, move };
                }).filter(Boolean);
                
                if (batch.length === 0) {
                    if (completedMoves === history.length) {
                        // All moves completed, clean up and resolve
                        workerPool.forEach(worker => {
                            if (worker.worker && worker.worker.terminate) {
                                worker.worker.terminate();
                            }
                        });
                        progressCallback(100);
                        resolve(moves);
                        return;
                    }
                    
                    // Wait for workers to finish if no available workers but queue not empty
                    setTimeout(processBatch, 100);
                    return;
                }
                
                // Process each position in the batch
                await Promise.all(batch.map(async ({ worker, move }) => {
                    try {
                        // Start both cloud and local evaluations in parallel
                        const cloudPromise = MoveEvaluator.tryCloudEvaluation(move.fen);
                        const localPromise = worker.evaluate(move.fen, depth);
                        
                        // Race between cloud and local evaluation
                        // Use Promise.allSettled to get both results regardless of success/failure
                        const [cloudResult, localResult] = await Promise.allSettled([
                            // Add a timeout to the cloud evaluation as a fallback
                            Promise.race([
                                cloudPromise,
                                new Promise(resolve => setTimeout(() => resolve(undefined), 2500))
                            ]),
                            localPromise
                        ]);
                        
                        let lines;
                        let engineName;
                        
                        // If cloud evaluation succeeded with good depth, use it
                        if (cloudResult.status === 'fulfilled' && 
                            cloudResult.value && 
                            cloudResult.value.length > 1 && 
                            cloudResult.value[0].depth >= depth) {
                            
                            // If local evaluation was started, abort it
                            worker.abort();
                            
                            lines = cloudResult.value;
                            engineName = "Stockfish Cloud";
                        } 
                        // Otherwise use local evaluation results
                        else if (localResult.status === 'fulfilled') {
                            lines = localResult.value;
                            engineName = worker.engine.name;
                        } 
                        // Fallback in case both failed
                        else {
                            lines = [];
                            engineName = "Failed Evaluation";
                        }
                        
                        move.lines = lines;
                        move.engine = engineName;
                        moves[move.i] = move; // Store at original index position
                    } catch (error) {
                        console.error("Error evaluating move:", error);
                        moves[move.i] = move; // Store it anyway without lines
                    } finally {
                        completedMoves++;
                        worker.busy = false;
                        
                        // Calculate and report progress
                        const progress = Math.round((completedMoves / history.length) * 100);
                        progressCallback(progress);
                    }
                }));
                
                // Process next batch
                processBatch();
            }
            
            // Start processing
            if (history.length === 0) {
                progressCallback(100);
                resolve([]);
            } else {
                processBatch();
            }
        });
    }

    /**
     * Analyzes a chess game and assigns move classifications
     * @param {Object} game - game object
     * @param {Function} progressCallback - Optional callback for progress updates
     * @returns {Promise<Array>} - Array of evaluated moves with classifications
     */
    static async analyzeGame(game, progressCallback = null, settings = {}) {
        const chess = new Chess();

        // Get the move list and reset to starting position
        chess.loadPgn(game.pgn);
        const history = chess.history();
        chess.reset();

        if (history.length === 0) {
            return {
                white: {
                    accuracy: 0,
                    counts: {}
                },
                black: {
                    accuracy: 0,
                    counts: {}
                },
                moves: []
            }
        }

        const moves = await MoveEvaluator.batchEvaluateMoves(chess, history, (progress) => {
            if (progressCallback) progressCallback(progress);
        }, settings);

        // Loop through moves and assign classifications
        moves[0].classification = Classification.THEORY;
        moves[0].graph = 50; // 0.0 eval is 50% eval bar basically
        for (let i = 1; i < moves.length; i++) {
            const move = moves[i];
            const previous = moves[i - 1];
            const movesUpToCurrent = moves.slice(0, i).map(m => m.move);

            MoveClassifier.classifyMove(move, previous, movesUpToCurrent);
        }

        MoveAnnotator.annotateMoves(moves, 'w');
        
        // Calculate accuracy
        const whiteMoves = moves.filter(move => move.fen.includes(' b '));
        const blackMoves = moves.filter(move => move.fen.includes(' w '));
        const whiteAccuracy = whiteMoves.reduce((sum, move) => sum + move.classification.accuracy, 0) / whiteMoves.length;
        const blackAccuracy = blackMoves.reduce((sum, move) => sum + move.classification.accuracy, 0) / blackMoves.length;

        // Calculate counts of each classification type
        const whiteCounts = whiteMoves.reduce((counts, move) => {
            counts[move.classification.type] = (counts[move.classification.type] || 0) + 1;
            return counts;
        }, {});
        const blackCounts = blackMoves.reduce((counts, move) => {
            counts[move.classification.type] = (counts[move.classification.type] || 0) + 1;
            return counts;
        }, {});

        // Calculate estimated Elo from centipawn loss
        const whiteCpl = whiteMoves.reduce((sum, move) => sum + move.centipawnLoss || 0, 0) / whiteMoves.length;
        const blackCpl = blackMoves.reduce((sum, move) => sum + move.centipawnLoss || 0, 0) / blackMoves.length;

        const getEloFromAverageCpl = (averageCpl) => {
            const cpl = Math.max(Math.min(averageCpl, 300), 0) / 100;
            return -958.03125 * Math.pow(cpl, 3) + 3395.605 * Math.pow(cpl, 2) - 4216.92255 * cpl + 2574.1609;
        }
        
        const getAverageCplFromElo = (elo) => {
            return 2.09951 * Math.pow(0.998681, Math.max(Math.min(elo, 2800), 100));
        }
        const getEloFromRatingAndCpl = (gameCpl, rating, backUpRating) => {
            if (!rating || rating === 'Unrated') rating = backUpRating;

            const eloFromCpl = getEloFromAverageCpl(gameCpl);
            if (!rating) return eloFromCpl;
            
            const expectedCpl = getAverageCplFromElo(rating);
            const cplDiff = gameCpl - expectedCpl;
            if (cplDiff === 0) return eloFromCpl;

            if (cplDiff > 0) {
                return rating * Math.exp(-0.005 * cplDiff);
            } else {
                return rating / Math.exp(-0.005 * -cplDiff);
            }
        };

        const phases = GamePhase.getPhases(moves.map(m => m.move));
        const hasEndgame = phases[1] !== undefined;
        
        // Helper function to calculate accuracy for a set of moves
        const calculateAccuracy = (movesArray) => {
            if (movesArray.length === 0) return 0;
            return movesArray.reduce((sum, move) => sum + move.classification.accuracy, 0) / movesArray.length;
        };
        
        // Helper function to determine classification based on accuracy and special moves
        const getClassification = (accuracy, hasBrilliant, hasGreat) => {
            if (accuracy > 0.8 && hasBrilliant) return Classification.BRILLIANT;
            if (accuracy > 0.8 && hasGreat) return Classification.GREAT;
            if (accuracy > 0.8) return Classification.PERFECT;
            if (accuracy > 0.7) return Classification.EXCELLENT;
            if (accuracy > 0.65) return Classification.GOOD;
            if (accuracy > 0.6) return Classification.MISTAKE;
            return Classification.BLUNDER;
        };
        
        // Split moves into phases
        const phaseMoves = {
            opening: moves.slice(0, phases[0].startMove),
            middlegame: moves.slice(phases[0].startMove, phases[1]?.startMove || moves.length),
            ...(hasEndgame && { endgame: moves.slice(phases[1].startMove) })
        };

        // Process each phase
        const phaseAnalysis = {};
        const phaseClassifications = { white: {}, black: {} };
        
        Object.entries(phaseMoves).forEach(([phaseName, phaseMovesArray]) => {
            const whiteMoves = phaseMovesArray.filter(m => m.fen.includes(' b '));
            const blackMoves = phaseMovesArray.filter(m => m.fen.includes(' w '));
            
            const whiteAccuracy = calculateAccuracy(whiteMoves);
            const blackAccuracy = calculateAccuracy(blackMoves);
            
            const whiteBrilliant = whiteMoves.some(m => m.classification.type === 'brilliant');
            const blackBrilliant = blackMoves.some(m => m.classification.type === 'brilliant');
            const whiteGreat = whiteMoves.some(m => m.classification.type === 'great');
            const blackGreat = blackMoves.some(m => m.classification.type === 'great');
            
            phaseAnalysis[phaseName] = {
                white: { accuracy: whiteAccuracy, brilliant: whiteBrilliant, great: whiteGreat },
                black: { accuracy: blackAccuracy, brilliant: blackBrilliant, great: blackGreat }
            };
            
            phaseClassifications.white[phaseName] = getClassification(whiteAccuracy, whiteBrilliant, whiteGreat);
            phaseClassifications.black[phaseName] = getClassification(blackAccuracy, blackBrilliant, blackGreat);
        });

        return {
            white: {
                accuracy: whiteAccuracy,
                counts: whiteCounts,
                elo: getEloFromRatingAndCpl(whiteCpl, game.white.elo, game.black.elo)
            },
            black: {
                accuracy: blackAccuracy,
                counts: blackCounts,
                elo: getEloFromRatingAndCpl(blackCpl, game.black.elo, game.white.elo)
            },
            phaseClassifications,
            moves: moves
        };
    }

    /**
     * Updates move tree nodes with classifications from analyzed moves
     * @param {MoveTree} moveTree - The move tree object
     * @param {Array} moves - Array of evaluated moves with classifications
     * @param {string} pgn - PGN string of the game
     */
    static applyClassificationsToMoveTree(moveTree, moves, pgn) {
        const game = new Chess();
        
        game.loadPgn(pgn);
        const history = game.history({ verbose: true });
        game.reset();
        
        // Apply classifications to move tree nodes
        for (let i = 0; i < moves.length; i++) {
            if (i < history.length) {
                const move = history[i];
                const moveId = `move_${Math.ceil((i+1)/2)}_${i % 2 === 0 ? 'w' : 'b'}_${move.san.replace('+', 'check').replace('#', 'mate')}`;
                moveTree.updateClassification(moveId, moves[i]);
                
                // Also store the evaluation score and type in the node
                const node = moveTree.nodeMap.get(moveId);
                if (node && moves[i].lines && moves[i].lines.length > 0) {
                    const topLine = moves[i].lines.find(line => line.id === 1);
                    if (topLine) {
                        node.evalScore = topLine.score;
                        node.evalType = topLine.type || 'cp';
                    }
                }
            }
        }
        
        return moveTree;
    }
}