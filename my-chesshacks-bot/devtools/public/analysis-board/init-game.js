// This should be loaded as a module after all dependencies are loaded
import { ChessUI } from "./src/components/ChessUI.js";
import { GameLoader } from "./src/components/games/GameLoader.js";
import { GameGraph } from "./src/components/report/GameGraph.js";
import { DebugInformation } from "./src/components/debug/DebugInformation.js";

window.initAnalysisBoard = async function (gameData) {
  const chessUI = new ChessUI();
  const game = GameLoader.loadGameFromPGN(gameData.pgn);

  game.white.name = gameData.white.name;
  game.black.name = gameData.black.name;
  game.white.elo = gameData.white.elo || "N/A";
  game.black.elo = gameData.black.elo || "N/A";
  game.username =
    gameData.playerColor === "black"
      ? gameData.black.name
      : gameData.white.name;

  $("#white-name").text(game.white.name);
  $("#black-name").text(game.black.name);
  $("#white-rating").text(game.white.elo);
  $("#black-rating").text(game.black.elo);

  await chessUI.load(game);

  if (gameData.mode === "play-vs-bot") {
    console.log("initAnalysisBoard: enabling play-vs-bot mode", {
      playerColor: gameData.playerColor
    });

    chessUI.onHumanMove(async () => {
      try {
        console.log("initAnalysisBoard: human move detected");
        const pgn =
          chessUI && typeof chessUI.getGamePGN === "function"
            ? chessUI.getGamePGN()
            : game.pgn;
        if (pgn && typeof pgn === "string") {
          game.pgn = pgn;
          if (chessUI.game) {
            chessUI.game.pgn = pgn;
          }
        }
        const timeleft =
          typeof gameData.timeleft === "number" ? gameData.timeleft : 60000;

        console.log("initAnalysisBoard: sending request to /api/bot");

        const res = await fetch("/api/bot", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ pgn, timeleft })
        });

        if (!res.ok) {
          console.error("initAnalysisBoard: /api/bot returned non-ok", {
            status: res.status
          });
          return;
        }

        const data = await res.json();
        if (!data || !data.move) {
          console.error(
            "initAnalysisBoard: /api/bot response missing move",
            data
          );
          return;
        }

        console.log("initAnalysisBoard: applying bot move", {
          move: data.move
        });
        const botMoveResult = chessUI.playUciMove(data.move);

        try {
          const currentNode = chessUI.moveTree?.currentNode;
          const prevNode = chessUI.moveTree?.getPreviousMove
            ? chessUI.moveTree.getPreviousMove()
            : null;

          if (
            currentNode &&
            data.move_probs &&
            typeof data.move_probs === "object"
          ) {
            currentNode.probabilities = data.move_probs;
          }

          if (currentNode && data.logs) {
            currentNode.logs = data.logs;
          }

          if (currentNode) {
            const showProbabilityArrows = chessUI.settingsMenu?.getSettingValue
              ? chessUI.settingsMenu.getSettingValue("showProbabilityArrows")
              : true;
            DebugInformation.updateDebugInfo(
              currentNode,
              prevNode,
              chessUI.board,
              showProbabilityArrows
            );
          }
        } catch (e) {
          console.error(
            "initAnalysisBoard: failed to apply move probabilities",
            e
          );
        }
      } catch (e) {
        console.error("initAnalysisBoard: error during bot move flow", e);
      }
    });
  }

  const tabClickHandler = function () {
    $(".tab-button").removeClass("active");
    $(".tab-panel").removeClass("active");
    $(this).addClass("active");

    const tabName = $(this).data("tab");
    $("#" + tabName + "-tab").addClass("active");

    GameGraph.render();
  };

  $(".tab-button").on("click", tabClickHandler);
  chessUI.addTeardownCallback(() =>
    $(".tab-button").off("click", tabClickHandler)
  );

  window.chessUI = chessUI;

  return chessUI;
};
