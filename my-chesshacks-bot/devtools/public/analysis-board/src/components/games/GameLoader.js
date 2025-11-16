export const Platform = {
    CHESSCOM: 'chesscom',
    LICHESS: 'lichess',
    PGN: 'pgn'
}

export class GameLoader {
    constructor() {}

    static parsePGN(pgn) {
        const data = {};
        const metadataRegex = /\[(\w+)\s+"([^"]+)"\]/g;
        let match;
        
        while ((match = metadataRegex.exec(pgn)) !== null) {
            data[match[1].toLowerCase()] = match[2];
        }
        
        return data;
    }

    static async fetchSingleLichessGame(gameId) {
        try {
            const response = await fetch(`https://lichess.org/game/export/${gameId}`, {
                headers: { 'Accept': 'application/x-chess-pgn' }
            });
            
            if (!response.ok) {
                throw new Error(`Failed to fetch Lichess game: ${response.status}`);
            }
            
            const pgn = await response.text();
            return {
                pgn: pgn.trim(),
                url: `https://lichess.org/${gameId}`
            };
        } catch (error) {
            console.error('Error fetching single Lichess game:', error);
            return null;
        }
    }

    static async fetchSingleChessComGame(username, gameId) {
        try {
            const archivesResponse = await fetch(`https://api.chess.com/pub/player/${username}/games/archives`);
            if (!archivesResponse.ok) throw new Error(`Failed to fetch archives: ${archivesResponse.status}`);
            
            const { archives = [] } = await archivesResponse.json();
            for (const archiveUrl of archives.reverse()) {
                try {
                    const archiveResponse = await fetch(archiveUrl);
                    if (!archiveResponse.ok) continue;
                    
                    const archiveData = await archiveResponse.json();
                    const game = archiveData.games?.find(g => g.url?.includes(gameId));
                    
                    if (game) {
                        return {
                            pgn: game.pgn,
                            url: game.url,
                            rated: game.rated
                        };
                    }
                } catch (error) {}
            }
            
            console.error(`Game ${gameId} not found in archives`);
        } catch (error) {
            console.error(`Error fetching single Chess.com game: ${error}`);
            return null;
        }
    }

    static async fetchSingleGame(username, gameId, platform = Platform.CHESSCOM) {
        try {
            if (platform === Platform.LICHESS) {
                return await this.fetchSingleLichessGame(gameId);
            } else {
                return await this.fetchSingleChessComGame(username, gameId);
            }
        } catch (error) {
            console.error(`Error fetching single game:`, error);
            return null;
        }
    }

    static async fetchPlayerAvatar(username, platform = Platform.CHESSCOM) {
        try {
            if (platform === Platform.LICHESS) {
                // Lichess doesn't have profile pictures
                return null;
            } else {
                const response = await fetch(`https://api.chess.com/pub/player/${username}`);
                if (response.ok) {
                    const data = await response.json();
                    return data.avatar;
                }
            }
        } catch (error) {
            console.warn(`Failed to fetch avatar for ${username}:`, error);
        }
        return null;
    }

    static async loadGameFromURL() {
        const params = new URLSearchParams(document.location.search);
        const username = params.get("user");
        const gameId = params.get("id");
        const platform = params.get("platform") || Platform.CHESSCOM;
        
        if (!username || !gameId) {
            return console.info('Missing username or gameId in URL parameters');
        }

        try {
            const game = await this.fetchSingleGame(username, gameId, platform);
            
            if (!game?.pgn?.trim()) {
                console.error(`Game not found or has no PGN: ${gameId}`);
                return this.loadEmptyGame();
            }

            console.info(`Successfully loaded game: ${gameId}`);

            const pgnData = this.parsePGN(game.pgn);

            // Lichess doesn't support profile avatars
            const avatarSupport = platform === Platform.CHESSCOM;
            const whiteAvatar = avatarSupport ? this.fetchPlayerAvatar(pgnData.white) : undefined;
            const blackAvatar = avatarSupport ? this.fetchPlayerAvatar(pgnData.black) : undefined;

            return {
                username: username,
                pgn: game.pgn,
                result: pgnData.result || '*',
                white: {
                    name: pgnData.white || 'Unknown',
                    elo: pgnData.whiteelo || 'Unrated',
                    avatar: whiteAvatar
                },
                black: {
                    name: pgnData.black || 'Unknown',
                    elo: pgnData.blackelo || 'Unrated',
                    avatar: blackAvatar
                },
            }
        } catch (error) {
            console.error(`Error loading game:`, error);
            return null;
        }
    }

    static loadEmptyGame() {
        return {
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
    }

    static loadGameFromPGN(pgn) {
        // See if we can pull the username from the pgn
        const pgnData = this.parsePGN(pgn);
        const white = pgnData.white || 'White';
        const black = pgnData.black || 'Black';

        const whiteElo = pgnData.whiteelo || 'Unrated';
        const blackElo = pgnData.blackelo || 'Unrated';

        const username = white || black;

        return {
            username: username,
            pgn: pgn,
            result: pgnData.result || '*',
            white: {
                name: white,
                elo: whiteElo
            },
            black: {
                name: black,
                elo: blackElo
            },
        }
    }
}