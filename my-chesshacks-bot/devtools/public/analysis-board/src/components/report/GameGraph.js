/**
 * Manages the rendering of the game evaluation graph
 */
export class GameGraph {
  static canvas = null;
  static ctx = null;
  static analysis = null;
  static hoverIndex = -1;
  static isHovering = false;
  static hoverX = 0;
  static hoverY = 0;
  static currentMove = null;
  static scaleFactor = 2;
  static initialized = false;
  static clickCallback = null;
  static eventHandlers = null;

  static get canvasElement() {
    if (!this.canvas) {
      this.canvas = document.getElementById("game-analysis-graph");
      this.ctx = this.canvas?.getContext("2d");
      this.setupEventListeners();
    }
    return this.canvas;
  }

  static handleMouseEnter() {
    if (!this.canvas) return;
    this.isHovering = true;
    this.canvas.style.cursor = "pointer";
    this.render();
  }

  static handleMouseLeave() {
    if (!this.canvas) return;
    this.isHovering = false;
    this.canvas.style.cursor = "default";
    this.render();
  }

  static setupEventListeners() {
    if (!this.canvas || this.initialized) return;

    if (!this.eventHandlers) {
      this.eventHandlers = {
        mousemove: this.handleMouseMove.bind(this),
        mouseenter: this.handleMouseEnter.bind(this),
        mouseleave: this.handleMouseLeave.bind(this),
        click: this.handleClick.bind(this),
        touchstart: this.handleClick.bind(this),
        resize: this.render.bind(this),
      };
    }

    this.canvas.addEventListener("mousemove", this.eventHandlers.mousemove);
    this.canvas.addEventListener("mouseenter", this.eventHandlers.mouseenter);
    this.canvas.addEventListener("mouseleave", this.eventHandlers.mouseleave);
    this.canvas.addEventListener("click", this.eventHandlers.click);
    this.canvas.addEventListener("touchstart", this.eventHandlers.touchstart);

    window.addEventListener("resize", this.eventHandlers.resize);
    this.initialized = true;
    this.render();
  }

  static setAnalysis(analysis) {
    this.analysis = analysis;
    this.render();
  }

  static setClickCallback(callback) {
    this.clickCallback = callback;
  }

  static handleMouseMove(event) {
    if (!this.analysis?.moves?.length) return;

    const rect = this.canvasElement.getBoundingClientRect();
    const x = (event.clientX - rect.left) * this.scaleFactor;
    const y = (event.clientY - rect.top) * this.scaleFactor;

    this.hoverX = x;
    this.hoverY = y;

    const width = this.canvasElement.width;
    const moves = this.analysis.moves;
    const total = moves.length;
    const increment = width / total;

    const moveIndex = Math.min(
      Math.max(0, Math.floor(x / increment)),
      total - 1,
    );

    if (this.hoverIndex !== moveIndex) {
      this.hoverIndex = moveIndex;
      this.render();
    }
  }

  static handleClick(event) {
    if (!this.analysis?.moves?.length || !this.clickCallback) return;

    const rect = this.canvasElement.getBoundingClientRect();
    const x = (event.clientX - rect.left) * this.scaleFactor;

    const width = this.canvasElement.width;
    const moves = this.analysis.moves;
    const total = moves.length;
    const increment = width / total;

    const moveIndex = Math.min(
      Math.max(0, Math.floor(x / increment)),
      total - 1,
    );
    const clickedMove = moves[moveIndex];

    if (clickedMove) {
      this.clickCallback(clickedMove);
    }
  }

  static updateCurrentMoveNumber(moveNumber) {
    this.currentMove = moveNumber;
    this.render();
  }

  static render() {
    if (!this.canvasElement || !this.ctx) return;

    const canvas = this.canvasElement;
    const ctx = this.ctx;

    const height = parseInt($(".game-graph").css("height")) * this.scaleFactor;
    const width = parseInt($(".game-graph").css("width")) * this.scaleFactor;

    canvas.width = canvas.clientWidth * this.scaleFactor;
    canvas.height = canvas.clientHeight * this.scaleFactor;

    // Draw background
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, width, height);

    if (
      !this.analysis ||
      !this.analysis.moves ||
      this.analysis.moves.length === 0
    ) {
      // Draw loading background
      ctx.fillStyle = "#dddddd";
      ctx.fillRect(0, height / 2, width, height);

      ctx.fillStyle = "grey";
      ctx.fillRect(0, height / 2 - 1, width, 2);
      return;
    }

    const moves = this.analysis.moves;
    const total = moves.length;

    if (total > 0) {
      const increment = width / total;

      ctx.beginPath();
      ctx.moveTo(-3, height);
      ctx.lineTo(0, height / 2);

      for (let i = 0; i < total; i++) {
        const move = moves[i];
        const offset = increment * i;
        const x = offset + 3;
        const y = (height / 100) * move.graph;

        ctx.lineTo(x, y);

        if (i === total - 1) {
          ctx.lineTo(increment * (total - 1) + 50, y);
        }
      }

      ctx.lineTo(increment * (total - 1) + 50, height);

      // Fill with white/gray
      ctx.fillStyle = "#dddddd";
      ctx.fill();

      // Draw the center line
      ctx.fillStyle = "#80808075";
      ctx.fillRect(0, height / 2 - 1, width, 2 * this.scaleFactor);

      if (this.currentMove) {
        const move = moves[2 * this.currentMove - 2];
        if (!move) return;

        const offset = increment * this.currentMove * 2 - increment * 2;
        ctx.fillStyle = "#99999975";
        ctx.fillRect(
          offset - this.scaleFactor,
          0,
          2 * this.scaleFactor,
          height,
        );

        const y = (height / 100) * move.graph;

        // Highlight dot
        ctx.fillStyle = move.classification.color;
        ctx.beginPath();
        ctx.arc(
          offset,
          y,
          4 * this.scaleFactor,
          0,
          Math.PI * 2 * this.scaleFactor,
        );
        ctx.fill();
      }

      // Draw hover effects
      if (this.isHovering && this.hoverIndex >= 0 && this.hoverIndex < total) {
        const move = moves[this.hoverIndex];

        const offset = increment * this.hoverIndex;
        const x = offset;
        const y = (height / 100) * move.graph;

        // Vertical line
        ctx.fillStyle = "#99999975";
        ctx.fillRect(x - this.scaleFactor, 0, 2 * this.scaleFactor, height);

        // Highlight dot
        ctx.fillStyle = move.classification.color;

        ctx.beginPath();
        ctx.arc(x, y, 4 * this.scaleFactor, 0, Math.PI * 2 * this.scaleFactor);
        ctx.fill();

        this.drawEvaluationPopup(move, x, y - 22 * this.scaleFactor);
      }
    }
  }

  static drawEvaluationPopup(move, x, y) {
    const ctx = this.ctx;
    const popupWidth = 40 * this.scaleFactor;
    const popupHeight = 22 * this.scaleFactor;

    let evalText;
    const bestLine = move.lines.find((line) => line.id === 1);
    if (bestLine) {
      const evalValue = bestLine.score / 100;
      const isMate = bestLine.type === "mate";
      evalText = isMate
        ? `M${Math.abs(bestLine.score)}`
        : evalValue > 0
          ? `+${evalValue.toFixed(2)}`
          : evalValue.toFixed(2);
    } else {
      evalText = "0-1";
    }

    // Keep popup within canvas bounds
    const adjustedX = Math.min(
      Math.max(x - popupWidth / 2, 10),
      this.canvasElement.width * this.scaleFactor - popupWidth - 10,
    );
    const adjustedY = Math.min(
      Math.max(y - popupHeight / 2, 50),
      this.canvasElement.height * this.scaleFactor - popupHeight - 10,
    );

    // Draw popup background
    ctx.fillStyle = "rgba(58, 58, 58, 0.95)";
    this.roundRect(ctx, adjustedX, adjustedY, popupWidth, popupHeight, 5, true);

    // Draw text
    ctx.fillStyle = "white";
    ctx.font = `bold ${12 * this.scaleFactor}px Arial`;
    ctx.textAlign = "center";
    ctx.fillText(
      `${evalText}`,
      adjustedX + popupWidth / 2,
      adjustedY + 15 * this.scaleFactor,
    );
  }

  static roundRect(ctx, x, y, width, height, radius, fill, stroke) {
    if (typeof radius === "undefined") {
      radius = 5;
    }

    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();

    if (fill) {
      ctx.fill();
    }

    if (stroke) {
      ctx.stroke();
    }
  }

  static destroy() {
    if (this.canvas && this.eventHandlers) {
      this.canvas.removeEventListener(
        "mousemove",
        this.eventHandlers.mousemove,
      );
      this.canvas.removeEventListener(
        "mouseenter",
        this.eventHandlers.mouseenter,
      );
      this.canvas.removeEventListener(
        "mouseleave",
        this.eventHandlers.mouseleave,
      );
      this.canvas.removeEventListener("click", this.eventHandlers.click);
      this.canvas.removeEventListener(
        "touchstart",
        this.eventHandlers.touchstart,
      );
      window.removeEventListener("resize", this.eventHandlers.resize);
    }

    this.canvas = null;
    this.ctx = null;
    this.analysis = null;
    this.hoverIndex = -1;
    this.isHovering = false;
    this.hoverX = 0;
    this.hoverY = 0;
    this.currentMove = null;
    this.initialized = false;
    this.clickCallback = null;
    this.eventHandlers = null;
  }
}
