export class GameClassifier {
  constructor() {
    // Thresholds for classification
    this.BLUNDER_THRESHOLD = 0.25; // Minimum evaluation swing to be considered a blunder
    this.DECISIVE_ADVANTAGE = 0.75; // When a position is considered winning
    this.EQUAL_RANGE = 0.1; // Range around 0.5 considered equal
    this.VOLATILITY_THRESHOLD = 0.15; // Average swing for "wild" games
  }

  classifyGame(evaluations, perspective = 'w', gameResult = '*') {
    if (!evaluations || evaluations.length < 2) {
      return { classification: 'insufficient_data', message: 'Please select a game to analyze in the games tab by uploading a PGN or searching games!' };
    }

    const analysis = this.analyzeGame(evaluations);
    const message = this.generateMessage(analysis, perspective, gameResult);
    
    return {
      classification: analysis.gameType,
      message: message,
      details: {
        blunders: analysis.blunders,
        gameLength: evaluations.length,
        finalEval: evaluations[evaluations.length - 1],
        avgVolatility: analysis.avgVolatility,
        timeInAdvantage: analysis.timeInAdvantage
      }
    };
  }

  analyzeGame(evals) {
    const swings = this.calculateSwings(evals);
    const blunders = this.findBlunders(evals, swings);
    const volatility = this.calculateVolatility(swings);
    const timeInAdvantage = this.calculateTimeInAdvantage(evals);
    const gamePhases = this.identifyGamePhases(evals);

    // Determine game type based on analysis
    let gameType = 'balanced';
    
    if (blunders.length >= 1 && blunders.some(b => b.magnitude >= this.BLUNDER_THRESHOLD * 1.5)) {
      gameType = 'blunder_decided';
    } else if (volatility >= this.VOLATILITY_THRESHOLD) {
      gameType = 'volatile';
    } else if (timeInAdvantage.white > 0.7 || timeInAdvantage.black > 0.7) {
      gameType = 'dominant';
    } else if (Math.abs(evals[evals.length - 1] - 0.5) < this.EQUAL_RANGE) {
      gameType = 'close_finish';
    }

    return {
      gameType,
      blunders,
      avgVolatility: volatility,
      timeInAdvantage,
      gamePhases,
      finalEval: evals[evals.length - 1]
    };
  }

  calculateSwings(evals) {
    const swings = [];
    for (let i = 1; i < evals.length; i++) {
      swings.push(evals[i] - evals[i - 1]);
    }
    return swings;
  }

  findBlunders(evals, swings) {
    const blunders = [];
    
    for (let i = 0; i < swings.length; i++) {
      const swing = Math.abs(swings[i]);
      if (swing >= this.BLUNDER_THRESHOLD) {
        const moveNumber = Math.floor((i + 1) / 2) + 1;
        const isWhiteMove = (i + 1) % 2 === 1;
        const blunderer = swings[i] > 0 ? 'black' : 'white';
        
        blunders.push({
          moveIndex: i + 1,
          moveNumber,
          isWhiteMove,
          blunderer,
          magnitude: swing,
          evalBefore: evals[i],
          evalAfter: evals[i + 1]
        });
      }
    }
    
    return blunders.sort((a, b) => b.magnitude - a.magnitude);
  }

  calculateVolatility(swings) {
    if (swings.length === 0) return 0;
    const avgSwing = swings.reduce((sum, swing) => sum + Math.abs(swing), 0) / swings.length;
    return avgSwing;
  }

  calculateTimeInAdvantage(evals) {
    let whiteAdvantage = 0;
    let blackAdvantage = 0;
    let equal = 0;

    evals.forEach(evaluation => {
      if (evaluation > 0.5 + this.EQUAL_RANGE) {
        whiteAdvantage++;
      } else if (evaluation < 0.5 - this.EQUAL_RANGE) {
        blackAdvantage++;
      } else {
        equal++;
      }
    });

    const total = evals.length;
    return {
      white: whiteAdvantage / total,
      black: blackAdvantage / total,
      equal: equal / total
    };
  }

  identifyGamePhases(evals) {
    const phases = [];
    let currentPhase = this.getPhaseType(evals[0]);
    let phaseStart = 0;

    for (let i = 1; i < evals.length; i++) {
      const newPhase = this.getPhaseType(evals[i]);
      if (newPhase !== currentPhase) {
        phases.push({
          type: currentPhase,
          start: phaseStart,
          end: i - 1,
          length: i - phaseStart
        });
        currentPhase = newPhase;
        phaseStart = i;
      }
    }

    // Add final phase
    phases.push({
      type: currentPhase,
      start: phaseStart,
      end: evals.length - 1,
      length: evals.length - phaseStart
    });

    return phases;
  }

  getPhaseType(evaluation) {
    if (evaluation > 0.5 + this.EQUAL_RANGE) return 'white_advantage';
    if (evaluation < 0.5 - this.EQUAL_RANGE) return 'black_advantage';
    return 'equal';
  }

  generateMessage(analysis, perspective, gameResult) {
    const { gameType, blunders, timeInAdvantage, finalEval } = analysis;
    const isWhite = perspective.toLowerCase() === 'w';
    
    // Determine actual game outcome based on result
    let outcome = 'draw'; // Default to draw
    let playerWon = false;
    
    if (gameResult === '1-0') { // White won
      outcome = 'white_wins';
      playerWon = !isWhite;
    } else if (gameResult === '0-1') { // Black won
      outcome = 'black_wins';
      playerWon = isWhite;
    } else if (gameResult === '1/2-1/2' || gameResult === '1/2' || gameResult.includes('1/2')) {
      outcome = 'draw';
      playerWon = false; // No one wins in a draw
    } else {
      // For ongoing games or unknown results, fall back to evaluation
      const whiteWon = finalEval > 0.5;
      playerWon = (isWhite && whiteWon) || (!isWhite && !whiteWon);
      outcome = whiteWon ? 'white_wins' : 'black_wins';
    }
    
    // Create a deterministic seed based on game characteristics
    const gameSeed = this.createGameSeed(analysis);
    
    switch (gameType) {
      case 'blunder_decided':
        return this.generateBlunderMessage(blunders, isWhite, outcome, playerWon, gameSeed);
      
      case 'volatile':
        return this.generateVolatileMessage(outcome, playerWon, gameSeed);
      
      case 'dominant':
        return this.generateDominantMessage(timeInAdvantage, isWhite, outcome, playerWon, gameSeed);
      
      case 'close_finish':
        return this.generateCloseMessage(outcome, playerWon, gameSeed);
      
      default:
        return this.generateBalancedMessage(outcome, playerWon, gameSeed);
    }
  }

  createGameSeed(analysis) {
    // Create a deterministic seed based on game characteristics
    const { blunders, avgVolatility, timeInAdvantage, finalEval } = analysis;
    
    // Combine multiple game characteristics to create a unique but deterministic seed
    let seed = Math.floor(finalEval * 1000); // Final evaluation
    seed += Math.floor(avgVolatility * 10000); // Volatility
    seed += Math.floor(timeInAdvantage.white * 100); // Time advantage
    seed += blunders.length * 17; // Number of blunders
    
    // If there are blunders, add the biggest blunder magnitude
    if (blunders.length > 0) {
      seed += Math.floor(blunders[0].magnitude * 1000);
    }
    
    return seed;
  }

  getSeededIndex(seed, arrayLength) {
    // Simple deterministic pseudo-random selection
    return Math.abs(seed) % arrayLength;
  }

  generateBlunderMessage(blunders, isWhite, outcome, playerWon, gameSeed) {
    const biggestBlunder = blunders[0];
    const blundererIsPlayer = (biggestBlunder.blunderer === 'white' && isWhite) || 
                             (biggestBlunder.blunderer === 'black' && !isWhite);
    
    if (outcome === 'draw') {
      if (blundererIsPlayer) {
        const playerBlunderDrawMessages = [
          "The game was fairly balanced, but you made a significant error that your opponent couldn't quite convert into a win.",
          "You had a mistake that gave your opponent a big advantage, but they weren't able to find the winning continuation.",
          "A critical error on your part created winning chances for your opponent, but the game still ended in a draw.",
          "You made a costly mistake, but your opponent failed to capitalize fully and the game ended even.",
          "Despite a significant blunder that could have been decisive, both players ended up sharing the point."
        ];
        return playerBlunderDrawMessages[this.getSeededIndex(gameSeed + 13, playerBlunderDrawMessages.length)];
      } else {
        const opponentBlunderDrawMessages = [
          "Your opponent made a critical mistake that gave you winning chances, but the position proved difficult to convert.",
          "A significant error by your opponent created opportunities, though neither player could secure the full point.",
          "Your opponent's blunder offered you excellent winning chances, but the game still ended in a draw.",
          "Despite your opponent's costly mistake, the resulting position led to a drawn outcome.",
          "Your opponent made a big error, but the complexity of the position allowed them to hold the draw."
        ];
        return opponentBlunderDrawMessages[this.getSeededIndex(gameSeed + 14, opponentBlunderDrawMessages.length)];
      }
    } else if (blundererIsPlayer) {
      const playerBlunderMessages = [
        `That was a relatively even game, but you made a critical mistake that your opponent was able to capitalize on.`,
        `The game was fairly balanced until you made an error that your opponent took advantage of.`,
        `A competitive match overall, but one significant mistake on your part shifted the game in your opponent's favor.`,
        `You were holding your own well, but a key error gave your opponent the opportunity they needed.`,
        `The position was roughly equal, but you made a mistake that your opponent was able to convert.`
      ];
      return playerBlunderMessages[this.getSeededIndex(gameSeed, playerBlunderMessages.length)];
    } else {
      const opponentBlunderMessages = [
        `That was a relatively even game, but your opponent made one critical mistake that you were able to capitalize on.`,
        `A fairly balanced contest until your opponent made an error that you converted well.`,
        `The game was competitive throughout, but your opponent's mistake allowed you to take control.`,
        `You stayed patient in an equal position, and when your opponent made an error, you took advantage of it.`,
        `A well-fought game that could have gone either way, but your opponent's mistake gave you the win.`
      ];
      return opponentBlunderMessages[this.getSeededIndex(gameSeed + 1, opponentBlunderMessages.length)];
    }
  }

  generateVolatileMessage(outcome, playerWon, gameSeed) {
    if (outcome === 'draw') {
      const drawVolatileMessages = [
        "What a wild game! The evaluation swung back and forth dramatically, but neither player could convert their chances.",
        "A rollercoaster of a game with constant momentum shifts that ultimately ended fairly in a draw.",
        "That was an exciting battle with lots of ups and downs - a draw was a fitting end to such a dynamic game.",
        "The advantage changed hands multiple times in that thrilling contest, with both players sharing the point.",
        "An unpredictable game full of twists and turns that deserved the shared result."
      ];
      return drawVolatileMessages[this.getSeededIndex(gameSeed + 11, drawVolatileMessages.length)];
    } else if (playerWon) {
      const playerWonVolatileMessages = [
        "That was a back-and-forth game with lots of ups and downs, but you managed to come out on top.",
        "The advantage swung back and forth throughout the game, but you emerged victorious in the end.",
        "A game with several momentum shifts where you pulled through for the win.",
        "The evaluation changed frequently in that dynamic game, but you found a way to win.",
        "An unpredictable game with multiple lead changes, but you managed to secure victory."
      ];
      return playerWonVolatileMessages[this.getSeededIndex(gameSeed + 2, playerWonVolatileMessages.length)];
    } else {
      const playerLostVolatileMessages = [
        "That was a back-and-forth game with twists and turns, though it didn't go your way in the end.",
        "A dynamic game where the position swung frequently, but your opponent managed to edge you out.",
        "A game with several momentum changes that didn't fall your way in the end.",
        "The advantage changed hands multiple times in that shifting game, but your opponent found the winning path.",
        "An unpredictable contest with multiple lead changes, though your opponent came out ahead."
      ];
      return playerLostVolatileMessages[this.getSeededIndex(gameSeed + 3, playerLostVolatileMessages.length)];
    }
  }

  generateDominantMessage(timeInAdvantage, isWhite, outcome, playerWon, gameSeed) {
    const playerAdvantageTime = isWhite ? timeInAdvantage.white : timeInAdvantage.black;
    
    if (outcome === 'draw') {
      if (playerAdvantageTime > 0.7) {
        const playerDominantDrawMessages = [
          "You had the upper hand for most of the game, but couldn't quite convert your advantage into a win.",
          "Despite controlling the game for long periods, your opponent managed to hold the draw.",
          "You maintained pressure throughout most of the game, but your opponent's defense proved resilient.",
          "Good positional play gave you a lasting advantage, though your opponent found a way to secure the draw.",
          "You dictated the tempo for most of the game, but the final result was a hard-fought draw."
        ];
        return playerDominantDrawMessages[this.getSeededIndex(gameSeed + 15, playerDominantDrawMessages.length)];
      } else {
        const opponentDominantDrawMessages = [
          "Your opponent had the initiative for most of the game, but you defended well to secure the draw.",
          "Despite being under pressure for long periods, you managed to hold your ground and earn a draw.",
          "Your opponent controlled much of the game, but you showed good defensive skills to split the point.",
          "A challenging game where your opponent had the upper hand, but your resilience earned you half a point.",
          "Your opponent maintained steady pressure, but you found the key defensive resources to draw."
        ];
        return opponentDominantDrawMessages[this.getSeededIndex(gameSeed + 16, opponentDominantDrawMessages.length)];
      }
    } else if (playerAdvantageTime > 0.7) {
      const playerDominantMessages = [
        "You controlled most of that game and maintained an advantage throughout.",
        "Good positional play - you established an edge early and kept it for most of the game.",
        "A solid performance where you took control and held it from start to finish.",
        "You dictated the pace of that game, maintaining pressure throughout.",
        "Nice game management - you built your advantage and converted it effectively."
      ];
      return playerDominantMessages[this.getSeededIndex(gameSeed + 4, playerDominantMessages.length)];
    } else {
      const opponentDominantMessages = [
        "Your opponent had the upper hand for most of the game and maintained their advantage well.",
        "Your opponent controlled the tempo from early on and you had limited opportunities to equalize.",
        "A challenging game where your opponent established an edge and kept it throughout.",
        "Your opponent maintained steady pressure, leaving you with fewer chances to create counterplay.",
        "Your opponent showed good positional understanding, building and maintaining their advantage."
      ];
      return opponentDominantMessages[this.getSeededIndex(gameSeed + 5, opponentDominantMessages.length)];
    }
  }

  generateCloseMessage(outcome, playerWon, gameSeed) {
    if (outcome === 'draw') {
      const drawCloseMessages = [
        "A hard-fought game! The position stayed balanced throughout and both players earned a well-deserved draw.",
        "A tight contest where neither player could gain the decisive advantage - a fair result.",
        "Both players fought well in this closely contested game that deservedly ended in a draw.",
        "The game remained even throughout, with both sides having their chances before agreeing to split the point.",
        "A competitive battle where the balance held - sometimes the most accurate result is a draw."
      ];
      return drawCloseMessages[this.getSeededIndex(gameSeed + 10, drawCloseMessages.length)];
    } else if (playerWon) {
      const playerWonCloseMessages = [
        "A close game! The position stayed even right until the end, but you managed to edge out the victory.",
        "A narrow margin of victory in a game that could have gone either direction.",
        "You found a way to win in a tightly contested position.",
        "A tight finish where small details mattered - you found the winning path.",
        "The position remained balanced throughout, but you made the key decisions when it counted."
      ];
      return playerWonCloseMessages[this.getSeededIndex(gameSeed + 6, playerWonCloseMessages.length)];
    } else {
      const playerLostCloseMessages = [
        "That was a close and well-fought game that could have gone either way until the end.",
        "A narrow loss in a game decided by small margins.",
        "Close to victory - that was a tight contest where small details made the difference.",
        "A well-fought game that remained even throughout, though it didn't go your way.",
        "The outcome was uncertain until the end, but your opponent just found the edge."
      ];
      return playerLostCloseMessages[this.getSeededIndex(gameSeed + 7, playerLostCloseMessages.length)];
    }
  }

  generateBalancedMessage(outcome, playerWon, gameSeed) {
    if (outcome === 'draw') {
      const drawBalancedMessages = [
        "A solid, well-played game by both sides that ended in a fair draw.",
        "Both players demonstrated good chess understanding in this balanced contest that deservedly ended even.",
        "A competitive and even game where both sides played well - the draw reflects the quality of play.",
        "That was a well-fought, balanced game where neither player could gain the decisive edge.",
        "An evenly-matched contest where both players had their moments and shared the point fairly."
      ];
      return drawBalancedMessages[this.getSeededIndex(gameSeed + 12, drawBalancedMessages.length)];
    } else if (playerWon) {
      const playerWonBalancedMessages = [
        "That was a well-balanced game where both sides had their chances. Good job converting your opportunities.",
        "A solid, evenly-matched contest where you made the most of your key moments.",
        "Both players created chances in that balanced game, but you converted when it mattered.",
        "An even and competitive game where you capitalized on your opportunities well.",
        "A balanced contest where both sides had their moments, but you executed your plans more effectively."
      ];
      return playerWonBalancedMessages[this.getSeededIndex(gameSeed + 8, playerWonBalancedMessages.length)];
    } else {
      const playerLostBalancedMessages = [
        "That was a solid, balanced game where both players fought well. Your opponent managed to convert their chances slightly better.",
        "A well-contested game with chances for both sides, but your opponent was more effective in the key moments.",
        "Both players created opportunities in that even contest, but your opponent converted theirs more effectively.",
        "A competitive and balanced game where you played well, but your opponent edged you out in execution.",
        "An evenly-matched game where both sides had their chances, but your opponent capitalized on theirs better."
      ];
      return playerLostBalancedMessages[this.getSeededIndex(gameSeed + 9, playerLostBalancedMessages.length)];
    }
  }
}