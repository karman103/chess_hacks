export const funFacts = [
  `The longest chess game theoretically possible is 5,949 moves. That's more than 3 moves!`,
  `As late as 1560, Castling was actually two seperate moves. You had to play R-KB1 on one move and K-KN1 on the next move.`,
  `The word "Checkmate" in Chess comes from the Persian phrase "Shah Mat," which means "the King is dead."`,
  `The Police raided a Chess Tournament in Cleveland in 1973, arrested the Tournament director and confiscated the Chess sets on charges of allowing gambling (cash prizes to winners) and possession of gambling devices (the Chess sets).`,
  `The longest official chess game lasted 269 moves (I. Nikolic - Arsovic, Belgrade 1989) and ended in a draw. That's at least 3 moves!`,
  `Initially, the Queen could only move one square at a time, diagonally. Later, she could move two squares at a time, diagonally. It wasn't until Reconquista Spain, with its powerful queen Isabella, that the Queen became the strongest piece on the board.`,
  `Chess pieces don't look more realistic because before the game reached Europe, it passed through the Islamic world. Islam forbids making statues of animals or people, so chess pieces became vague-looking!`,
  `In many languages, the Pawn is a foot soldier, but in German and Spanish, it's a peasant or farmer, instead!`,
  `About six hundred million (600,000,000) people know how to play chess worldwide! That's over 3 people!`,
  `Chess is often cited by psychologists as an effective way to improve memory function.`,
  `From the starting position, there are eight different ways to Mate in two moves and 355 different ways to Mate in three moves.`,
  `Shogi legend Yoshiharu Habu took up chess as a hobby in his twenties. He played his first FIDE tournament at age 31 and finished with a 2342 rating.`,
  `The maximum number of Knights you can place on a chessboard without any attacking another is 32.`,
  `There are more possible chess games than there are atoms in the universe!`,
  `Frank James Marshall waited 8 years to test the Marshall Attack over the board. He finally unleashed it in 1918 against Capablanca, and lost.`,
  `The double pawn move, where pawns were allowed to advance two squares on their first move instead of one, was first introduced in Spain in 1281.`,
  `The first chess game played between space and earth was on June 9, 1970 by the Soyez-9 crew. The game ended in a draw.`,
  `The Ponomariov vs Fritz game on November 21, 2005 is the last known win by a human against a top performing computer under normal chess tournament conditions.`,
  `In 1956, at only 13 years old, Bobby Fischer beat Donald Byrne (age 26), an international chess champion at the time, in what became known as the "Game of the Century."`,
  `American-Icelandic chess player Bobby Fischer was one of the youngest grandmasters in history, earning the title at age 15.`,
  `Magnus Carlsen is the first chess champion to win in all three categories—blitz, regular, and rapid—in the same year.`,
  `Devised in the 19th century by Otto Blathy, the longest known chess problem ever created takes 290 moves to get to checkmate.`,
  `The first mechanical clock to be used instead of sand glass was invented by Thomas Wilson in 1883. The modern push button clock was introduced by Veenhoff in 1900.`,
  `The folding Chess board was invented in 1125 by a Chess-playing priest. Since the Church forbids priests to play Chess, he hid his Chessboard by making it look like two books stacked together.`,
  `The first computer program for playing chess was developed in 1951, by Alan Turing. However, no computer was powerful enough to process it, so Turing tested it by doing the calculations himself and playing according to the results, taking several minutes per move.`,
  `The first Chessboard with alternating light and dark squares appeared in Europe in 1090!`,
  `The second book ever printed in the English language was about chess!`,
  `The rook is named from an Arabic word rukh, meaning chariot. This reflects its ability to move quickly in straight lines, but not leap over obstacles.`,
  `The "Rook" in chess is often misattributed as the source of the term rookie, a first-year professional player. Although not certain, it is believed to be derived from the the word "recruit".`,
  `The first defeat of a reigning world chess champion by a computer under tournament conditions happened in Game 6 of the ACM Chess Challenge when Deep Blue played against Kasparov and won.`,
  `"I spend hours playing chess because I find it so much fun. The day it stops being fun is the day I give up." - Magnus Carlsen`,
  `"There are two types of sacrifices: Correct ones, and mine." - Mikhail Tal`,
  `"If your opponent offers you a draw, try to work out why he thinks he's worse off." - Nigel Short`,
  `"Of course, errors are not good for a chess game, but errors are unavoidable and in any case, a game without errors, or as they say 'flawless game' is colorless." - Mikhail Tal`,
  `During the 1972 Fischer-Spassky match in Rekjavik, the Russians linked Spassky's erratic play with Fischer's chair. The Icelandic organization put a 24-hour Police guard around the chair while chemical and x-ray tests were performed on the chair. Nothing unusual was found.`,
  `Armenia is the first country to make chess a mandatory subject in schools, aiming to foster cognitive skills and strategic thinking among students.`,
  `In German, the knight is called 'Springer', which translates to 'jumper'.`,
  `The village of Marottichal in Kerala, India, was transformed by chess. With over two-thirds of its 6,000 residents proficient in the game, it changed from a place plagued by alcoholism to a community passionate about the game.`,
  `World Chess Champion Magnus Carlsen demonstrated his exceptional memory by challenging contestants on BBC2's 'Chess Masters: The Endgame' to recall positions from his past games.`,
  `A hybrid sport called chess boxing combines chess and boxing, where participants alternate between rounds of chess and boxing, testing both mental and physical endurance.`,
  `Some chess masters can play multiple games simultaneously without sight of the boards, relying solely on memory and visualization skills, a feat known as blindfold chess.`,
  `In 1960, Hungarian Grandmaster Janos Flesch set a remarkable record by playing 52 opponents simultaneously while blindfolded. He won 31 of these games, showcasing an extraordinary memory and visualization ability.`,
  `There are over 1800 grandmasters in the world! You'll never be one of them.`,
  `"Even a poor plan is better than no plan at all." - Mikhail Chigorin`,
  `The first chess computer, The Mechanical Turk, was actually just a person hidden in a clever compartment.`,
  `In 1951, IM Robert Wade played a simultaneous exhibition against 30 Russian schoolboys. Wade won 0 games, drew 10, and lost the other 20.`,
  `I'm running out of fun facts, so please just pretend this was something really profound.`,
  `If you have a fun fact, please send it to me! Please. I need more.`,
];

export class SidebarOverlay {
  static overlay = null;
  static factIntervalId = null;
  static isAnalysisOverlayActive = false;

  static get $overlay() {
    if (!this.overlay) {
      this.overlay = $(".sidebar-overlay");
    }
    return this.overlay;
  }

  static show() {
    this.$overlay.addClass("active");
  }

  static hide() {
    this.$overlay.removeClass("active");
  }

  /**
   * Start cycling through chess facts on a timer
   */
  static startFactCycling() {
    this.stopFactCycling();

    this.factIntervalId = setInterval(() => {
      const facts = funFacts;
      if (!facts?.length) return;

      const $factElement = $(".fun-fact");
      if (!$factElement.length) return;

      // Get a different random fact
      const currentText = $factElement.text();
      let newFact;
      do {
        newFact = facts[Math.floor(Math.random() * facts.length)];
      } while (newFact === currentText && facts.length > 1);

      // Fade transition
      $factElement.animate({ opacity: 0 }, 300, function () {
        $(this).text(newFact).animate({ opacity: 1 }, 300);
      });
    }, 7107); // The average fact has 23.69 words, and most people can read at 200 wpm
  }

  /**
   * Stop cycling through chess facts
   */
  static stopFactCycling() {
    if (this.factIntervalId) {
      clearInterval(this.factIntervalId);
      this.factIntervalId = null;
    }
  }

  static reset() {
    this.stopFactCycling();
    this.isAnalysisOverlayActive = false;

    if (this.overlay) {
      this.overlay.removeClass("active");
    }
    this.overlay = null;

    $(".analysis-overlay, .board-overlay").removeClass("active");
    $(".tab-content, .bottom-content").removeClass("blur-content");
    $(".analysis-progress-bar").css("width", "0%");
    $(".progress-percentage").text("");
  }

  /**
   * Updates the evaluation progress bar
   * @param {number|object} progress - Progress percentage (0-100) or progress object with depth info
   */
  static updateEvaluationProgress(progress) {
    const percentage =
      typeof progress === "object" ? progress.percent || 0 : progress;

    // Show overlay when analysis starts
    if (percentage <= 5 && !this.isAnalysisOverlayActive) {
      this.isAnalysisOverlayActive = true;
      $(".analysis-overlay, .board-overlay").addClass("active");
      $(".tab-content, .bottom-content").addClass("blur-content");

      // Initialize progress bar and show random fact
      $(".analysis-progress-bar").css("width", percentage + "%");
      const facts = funFacts;
      if (facts?.length) {
        $(".fun-fact").text(facts[Math.floor(Math.random() * facts.length)]);
        this.startFactCycling();
      }
    }

    // Update progress during analysis
    if (this.isAnalysisOverlayActive) {
      $(".analysis-progress-bar").css("width", `${percentage}%`);

      let progressText = `${Math.round(percentage)}% complete`;
      if (typeof progress === "object" && progress?.depth) {
        progressText += ` (Depth ${progress.depth}/${progress.targetDepth})`;
      }
      $(".progress-percentage").text(progressText);
    }

    // Hide overlay when complete
    if (percentage >= 100 && this.isAnalysisOverlayActive) {
      setTimeout(() => {
        $(".analysis-overlay, .board-overlay").removeClass("active");
        $(".tab-content, .bottom-content").removeClass("blur-content");
        this.isAnalysisOverlayActive = false;
        this.stopFactCycling();
      }, 5);
    }
  }
}
