export class EvaluationBar {
    constructor() {}

    static scoreToPercentage = (score) => {
        score = Math.max(-3000, Math.min(3000, score));
    
        const sigmoid = (x) => 1 / (1 + Math.exp(-x / 300));
        let percentage = 50 + 45 * (2 * sigmoid(score) - 1);
    
        return Math.max(5, Math.min(95, percentage));
    }

    static updateEvaluationBar(evaluation = { evalScore: 0, evalType: 'cp' }) {
        const { evalScore, evalType } = evaluation;
        const isMate = evalType === 'mate';
        const isWhiteWinning = evalScore >= 0 && (!evaluation.mateForBlack);

        const percentage = isMate ? (isWhiteWinning ? 100 : 0) : this.scoreToPercentage(evalScore);
        const evalText = isMate ? `M${Math.abs(evalScore)}` : (Math.abs(evalScore) > 999) ? (Math.abs(evalScore) / 100).toFixed(0) : (Math.abs(evalScore) / 100).toFixed(1);

        // Update bar fill
        $(".eval-fill")
            .removeClass("white-advantage black-advantage")
            .addClass(isWhiteWinning ? "black-advantage" : "white-advantage")
            .stop().animate({ height: `${percentage}%` }, 500, 'swing');

        // Update/create eval text
        $(".eval-text").removeClass("white-winning black-winning")
            .addClass(isWhiteWinning ? "white-winning" : "black-winning")
            .text(evalText);
    }
}