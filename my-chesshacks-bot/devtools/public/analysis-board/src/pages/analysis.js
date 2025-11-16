import { GameLoader } from '../components/games/GameLoader.js';
import { ChessUI } from '../components/ChessUI.js';
import { GameGraph } from '../components/report/GameGraph.js';

async function loadPlayerData(white, black) {
    if (!white || !black) return;
    if (!white.name || !black.name) return;

    $("#white-name").text(white.name);
    $("#black-name").text(black.name);

    if (!white.elo || !black.elo) return;

    $("#white-rating").text(white.elo);
    $("#black-rating").text(black.elo);

    if (!white.avatar || !black.avatar) return;

    const whiteAvatar = await white.avatar;
    const blackAvatar = await black.avatar;

    $('#white-profile').attr('src', whiteAvatar);
    $('#black-profile').attr('src', blackAvatar);
}

const testgame = {
    username: 'White',
    pgn: '',
    result: '*',
    white: {
        name: 'White',
        elo: 'Unrated'
    },
    black: {
        name: 'Black',
        elo: 'Unrated'
    },
}

const testpgn = `[Event "Live Chess"]
[Site "Chess.com"]
[Date "2024.04.07"]
[Round "-"]
[White "jonniest"]
[Black "thankeo"]
[Result "1-0"]

[WhiteElo "1701"]
[BlackElo "1406"]
[TimeControl "600"]
[Termination "jonniest won by checkmate"]
[StartTime "04:27:31"]
[EndDate "2024.04.07"]
[EndTime "04:31:12"]
[Link "https://www.chess.com/game/live/106203025973"]

1. e4 {[%clk 0:09:57.7]} 1... b6 {[%clk 0:09:59]} 
2. Nf3 {[%clk 0:09:49.7]} 2... Bb7 {[%clk 0:09:58.1]} 
3. Nc3 {[%clk 0:09:45.1]} 3... Nc6 {[%clk 0:09:54]} 
4. d4 {[%clk 0:09:41.6]} 4... g6 {[%clk 0:09:52.9]} 
5. d5 {[%clk 0:09:36.9]} 5... Nb4 {[%clk 0:09:49.3]} 
6. a3 {[%clk 0:09:32.5]} 6... Na6 {[%clk 0:09:47.5]} 
7. Qd4 {[%clk 0:09:28.6]} 7... Nf6 {[%clk 0:09:44.8]} 
8. e5 {[%clk 0:09:24.5]} 8... Nh5 {[%clk 0:09:34.6]} 
9. Bxa6 {[%clk 0:09:18.6]} 9... Bxa6 {[%clk 0:09:32.7]} 
10. g4 {[%clk 0:09:13.9]} 10... Ng7 {[%clk 0:09:21]} 
11. Bh6 {[%clk 0:09:13.5]} 11... e6 {[%clk 0:09:17.3]} 
12. Qf4 {[%clk 0:09:10]} 12... c5 {[%clk 0:09:00.8]} 
13. Ng5 {[%clk 0:09:09.9]} 13... f5 {[%clk 0:08:44.2]} 
14. exf6 {[%clk 0:09:06]} 14... Nf5 {[%clk 0:08:24.4]} 
15. f7+ {[%clk 0:08:31.1]} 15... Ke7 {[%clk 0:08:13.6]} 
16. d6+ {[%clk 0:08:29.6]} 16... Kf6 {[%clk 0:08:07.6]} 
17. Nce4# {[%clk 0:08:23.6]} 1-0`

// Initialize the application when the document is ready
$(document).ready(async () => {
    const chessUI = new ChessUI();
    let game;

    game = GameLoader.loadGameFromPGN(testpgn);

    chessUI.load(game);
    loadPlayerData(game.white, game.black);

    // Tab switching
    $('.tab-button').on('click', function () {
        $('.tab-button').removeClass('active');
        $('.tab-panel').removeClass('active');
        $(this).addClass('active');

        const tabName = $(this).data('tab');
        $('#' + tabName + '-tab').addClass('active');

        GameGraph.render();
    });
});

