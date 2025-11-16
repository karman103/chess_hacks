import { Classification } from "./MoveClassifier.js";

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

// Currently in development, not used yet.
export class MoveAnnotator {
    constructor() {}

    static winComments = {
        you: [
            "Congratulations on the win!",
            "Great job securing the victory!",
            "Impressive play—well earned win!",
            "You outplayed your opponent. Nicely done!",
            "Victory is yours! Your strategy paid off.",
            "Excellent tactics led to a solid win!",
            "You kept your cool and it paid off—congrats!",
            "Smart moves and steady play. Well deserved!",
        ],
        opponent: [
            "That's a tough loss.",
            "You gave it a good fight—keep at it!",
            "Every game is a learning opportunity.",
            "Don't let this discourage you—you played well!",
            "A setback, but you're improving every time.",
            "Tough game, but there's always the next one!",
            "Even in defeat, there's something to gain.",
            "Keep pushing—resilience leads to mastery.",
        ]
    };

    static greatMoveComments = {
        you: [
            "This is the only move that works here, nice find!",
            "Impressive! Only one move worked here and you found it.",
            "You found the precise move—great calculation!",
            "Excellent! That was the only correct move in the position.",
            "Well done—any other move would have failed.",
        ],
        opponent: [
            "There was only one good move here, and they found it.",
            "They played the only move that holds—impressive.",
            "That was the only way to survive, and they spotted it.",
            "They found the narrow path—good defense.",
            "Only one correct move, and your opponent nailed it.",
        ]
    };

    static drawComments = [];
    static perfectComments = [];
    static goodComments = [];
    static mistakeComments = [];
    static blunderComments = {
        you: {
            throwaway: [],
            missed: [],
            other: []
        },
        opponent: {
            throwaway: [],
            missed: [],
            other: []
        },
    };

    static forcedComments = {
        you: {
            winning: [
                "No worries here, you're still winning.",
                "You're still winning, so no worries.",
                "The situation is fine here. You're winning.",
                "You're in control—even with this response.",
                "Even though it's forced, you're ahead.",
                "You had to play this, but the position favors you.",
                "It's a forced move, but your advantage remains intact.",
            ],
            losing: [
                "This was your only option.",
                "It's not good to let your opponent play forcing moves.",
                "You were backed into a corner here.",
                "Only one move kept you in the game — not ideal.",
                "Your opponent dictated the pace here.",
                "This forced response shows how much pressure you're under.",
                "That was the only move to avoid immediate disaster.",
                "You're surviving, but only just—forced and passive.",
                "Your hand was forced—there was no better alternative.",
                "Being limited to one response shows how strong their position is.",
            ]
        },
        opponent: {
            winning: [
                "Your opponent is forced to play this move, nice job!",
                "Your opponent's options are limited to playing this move—great control!",
                "That's a strong sequence—your opponent had no choice but to play this.",
                "Your opponent's only move shows how dominant your position is.",
                "Their only move here shows your strong initiative. Nice!",
            ],
            losing: [
                "Your opponent might be forced to play this, but you're losing.",
                "Even though you're dictating here, the overall position is bad.",
                "This move was forced, but it doesn't help you — you're still losing."
            ]
        }
    };

    static gotMatedComments = {
        you: {
            standard: [
                "This move allows you to be mated in %s moves.",
                "Be careful—this leads to mate in %s moves.",
                "Unfortunately, that blunder gives your opponent a forced mate in %s.",
                "This slip creates a mating sequence in %s moves.",
                "Oops! That move walks right into mate in %s.",
            ],
            missed: [
                "but your opponent missed it.",
                "luckily, your opponent didn't see it.",
                "but they overlooked the mating sequence.",
                "fortunately, they didn't capitalize on it.",
                "but the chance to finish it was missed.",
            ],
            hard: [
                "no worries, it's tricky to spot.",
                "that's a tough pattern—don't sweat it.",
                "even strong players miss that one.",
                "this kind of tactic is easy to overlook.",
                "it's a complex sequence—easy to miss.",
            ],
        },
        opponent: {
            standard: [
                "Your opponent blunders mate in %s moves.",
                "They just walked into a mate in %s.",
                "A critical mistake—mate in %s is possible now.",
                "Your opponent left themselves open to mate in %s moves.",
                "That move allows a forced mate in %s.",
            ],
            missed: [
                "but you missed it.",
                "and the opportunity slipped by.",
                "but you didn't spot the finish.",
                "unfortunately, you overlooked the tactic.",
                "but the mate was missed this time.",
            ],
            hard: [
                "and you spotted the winning combination!",
                "nicely done finding the mate!",
                "you saw the sequence—excellent work!",
                "great vision to capitalize on the blunder!",
                "you found the tactic—well played!",
            ],
        }
    };

    static brilliantComments = {
        you: {
            active: [
                "The sacrifice of your %s on %s is a devastating tactic!",
            ],
            passive: [
                "This move abandons your %s on %s, a brilliant sacrifice!",
            ]
        },
        opponent: {
            active: [
                "Your opponent finds the brilliant sacrifice %s on %s!",
            ],
            passive: [
                "This move permits you to capture the %s on %s, a brilliant sacrifice!",
            ]
        }
    };

    static pickRand(comments) {
        return comments[Math.floor(Math.random() * comments.length)];
    }

    static wasMatePlayed(moveIndex, moves) {
        const currentMove = moves[moveIndex];
        const opponentMove = moves[moveIndex + 1];
        if (!opponentMove) return true;

        const matingLines = currentMove.lines.filter(line => line.type === 'mate');
        if (matingLines.length === 0) return false;

        // Check if the opponent's move was one of the mating moves.
        const topMatingMoves = matingLines.map(line => line.pv[0]);
        const opponentPlayedMatingMove = topMatingMoves.includes(opponentMove.uciMove);

        // Check if the position after the opponent's move is still a mate.
        const opponentBestLine = opponentMove.lines.find(l => l.id === 1);
        const keepsMate = opponentBestLine && opponentBestLine.type === 'mate';

        // If they don't play a top mating line or they lose the mate, then it wasn't played.
        return opponentPlayedMatingMove && keepsMate;
    }

    static gotMated(move, moves, userPerspective) {
        const data = move.commentData;
        const mateIn = Math.abs(data.mateIn);
        const moveIndex = moves.findIndex(m => m.uciMove === move.uciMove && m.fen === move.fen);
        const wasMatePlayed = this.wasMatePlayed(moveIndex, moves);
        const isYou = userPerspective ? 'you' : 'opponent';

        const missed = this.pickRand(this.gotMatedComments[isYou].missed);
        const standard = this.pickRand(this.gotMatedComments[isYou].standard).replace("%s", mateIn);
        
        if (!wasMatePlayed) return standard.replace(".", ", " + missed);
        
        if (mateIn <= 5) return standard;
        return standard.replace(".", ", " + this.pickRand(this.gotMatedComments[isYou].hard));
    }

    static forced(move, userPerspective) {
        const isWinning = (move.win > 57 === userPerspective) ? 'winning' : 'losing';
        const isYou = userPerspective ? 'you' : 'opponent';
        
        return this.pickRand(this.forcedComments[isYou][isWinning]);
    }

    static brilliant(move, userPerspective) {
        const isYou = userPerspective ? 'you' : 'opponent';
        const sacrificedPieces = move.commentData.piecesViablyCapturable;

        // Map piece types to readable names
        const pieceNames = {
            'p': 'pawn',
            'r': 'rook', 
            'n': 'knight',
            'b': 'bishop',
            'q': 'queen',
            'k': 'king'
        };

        if (!sacrificedPieces || sacrificedPieces.length === 0) {
            return "Brilliant move!";
        }

        // Use the first sacrificed piece for the template
        const piece = sacrificedPieces[0];
        const pieceName = pieceNames[piece.type.toLowerCase()] || piece.type;

        // If the sacrificed piece is the piece we just moved, then it's an active move.
        if (move.uciMove.slice(2, 4) == piece.square) {
            return this.pickRand(this.brilliantComments[isYou].active).replace('%s', pieceName).replace('%s', piece.square);
        }

        // If the sacrificed piece is not the piece we just moved, then it's a passive move.
        return this.pickRand(this.brilliantComments[isYou].passive).replace('%s', pieceName).replace('%s', piece.square);
    }

    static annotateMove(move, moves, userPerspective) {
        const classification = move.classification;
        const type = move.commentType;
        const data = move.commentData;

        if (type === CommentType.WON) {
            return this.pickRand(this.winComments[userPerspective ? 'you' : 'opponent']);
        }

        if (type === CommentType.GOT_MATED) {
            return this.gotMated(move, moves, userPerspective);
        }

        if (type === CommentType.FORCED) {
            return this.forced(move, userPerspective);
        }

        if (classification === Classification.BRILLIANT) {
            return this.brilliant(move, userPerspective);
        }

        if (classification === Classification.GREAT) {
            return "This is the only move that works here, nice find!";
        }

        return "Default comment!"
    }

    static annotateMoves(moves, perspective) {
        const isWhite = perspective === 'w';
        // for (const move of moves) {
        //     move.comment = this.annotateMove(move, moves, isWhite);
        // }

        for (let i = 0; i < moves.length; i++) {
            const move = moves[i];
            
            // Is this move from the user's perspective?
            let userPerspective = i % 2 != 0;
            if (isWhite) userPerspective = !userPerspective;

            move.comment = this.annotateMove(move, moves, userPerspective);
        }
    }
}