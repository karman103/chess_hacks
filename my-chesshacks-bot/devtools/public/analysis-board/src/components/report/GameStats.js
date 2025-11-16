import { Classification } from '../../classification/MoveClassifier.js';

/**
 * GameStats utility class for displaying player statistics comparison
 */
export class GameStats {
    /**
     * Renders the game statistics UI to the specified container
     * @param {jQuery|string} container - The container element or selector
     * @param {Object} analysis - The game analysis data
     * @param {string} whiteName - White player name
     * @param {string} blackName - Black player name
     */
    static render(container = '.game-stats', analysis, whiteName = 'White', blackName = 'Black') {
        if (!analysis) {
            // Empty analysis
            analysis = {
                white: {
                    accuracy: 0,
                    counts: {},
                    elo: 0
                },
                black: {
                    accuracy: 0,
                    counts: {},
                    elo: 0
                },
                phaseClassifications: {
                    white: {
                        opening: Classification.GOOD,
                        middlegame: Classification.GOOD,
                        endgame: Classification.GOOD
                    },
                    black: {
                        opening: Classification.GOOD,
                        middlegame: Classification.GOOD,
                        endgame: Classification.GOOD
                    }
                }
            }
        }

        const $container = $(container);
        $container.empty();
        
        const statsContainer = $('<div class="stats-comparison"></div>');
        
        const phaseClassifications = analysis.phaseClassifications;

        // Add all sections
        statsContainer
            .append(this.createPlayersHeader(whiteName, blackName))
            .append('<hr class="stats-divider">')
            .append(this.createStatsRow('Accuracy', 
                (analysis.white.accuracy * 100).toFixed(1), 
                (analysis.black.accuracy * 100).toFixed(1), 
                true))
            .append('<hr class="stats-divider">')
            .append(this.createMovesSection(analysis))
            .append('<hr class="stats-divider">')
            .append(this.createStatsRow('Estimated Elo', 
                analysis.white.elo ? Math.ceil(analysis.white.elo / 10) * 10 : 1400, 
                analysis.black.elo ? Math.ceil(analysis.black.elo / 10) * 10 : 1400, 
                true))
            .append(this.createPhaseClassificationsRow('Opening', phaseClassifications?.white?.opening, phaseClassifications?.black?.opening, false))
            .append(this.createPhaseClassificationsRow('Middlegame', phaseClassifications?.white?.middlegame, phaseClassifications?.black?.middlegame, false))
            .append(this.createPhaseClassificationsRow('Endgame', phaseClassifications?.white?.endgame, phaseClassifications?.black?.endgame, false));



        $container.append(statsContainer);
    }

    /**
     * Creates the players header row
     * @param {string} whiteName - White player name
     * @param {string} blackName - Black player name
     * @returns {jQuery} The players header element
     */
    static createPlayersHeader(whiteName, blackName) {
        return $(`<div class="stats-row">
            <div class="stats-label"></div>
            <div class="stats-player">${whiteName}</div>
            <div class="stats-icon"></div>
            <div class="stats-player">${blackName}</div>
        </div>`);
    }

    /**
     * Creates a statistics row
     * @param {string} label - The row label
     * @param {string} whiteValue - White player value
     * @param {string} blackValue - Black player value
     * @param {boolean} useBoxes - Whether to use boxes for styling
     * @returns {jQuery} The stats row element
     */
    static createStatsRow(label, whiteValue, blackValue, useBoxes = false) {
        const whiteEl = useBoxes 
        ? `<div class="stats-box white">${whiteValue}</div>` 
        : `<div class="stats-count">${whiteValue}</div>`;
    
        const blackEl = useBoxes 
            ? `<div class="stats-box black">${blackValue}</div>` 
            : `<div class="stats-count"></div>`;

        return $(`<div class="stats-row">
            <div class="stats-label">${label}</div>
            ${whiteEl}
            <div class="stats-icon"></div>
            ${blackEl}
        </div>`);
    }

    static createPhaseClassificationsRow(label, whiteClassification, blackClassification) {
        const row = $(`<div class="stats-row">
            <div class="stats-label">${label}</div>
            <div id="white-icon" class=" stats-count"></div>
            <div class="stats-icon"></div>
            <div id="black-icon" class=" stats-count"></div>
        </div>`);

        let whiteIcon = $(`<p class="stats-label no-padding">-</p>`);
        let blackIcon = $(`<p class="stats-label no-padding">-</p>`);

        if (whiteClassification) whiteIcon = $(`<img src="${whiteClassification?.src || '-'}" alt="White" class="stats-icon">`);
        if (blackClassification) blackIcon = $(`<img src="${blackClassification?.src || '-'}" alt="Black" class="stats-icon">`);

        row.find('#white-icon').append(whiteIcon);
        row.find('#black-icon').append(blackIcon);

        return row;
    }

    /**
     * Creates the moves section with classification counts
     * @param {Object} analysis - The game analysis data
     * @returns {jQuery} The moves section element
     */
    static createMovesSection(analysis) {
        const movesSection = $('<div class="stats-moves"></div>');

        const displayedClassifications = [
            Classification.BRILLIANT, 
            Classification.GREAT, 
            Classification.PERFECT, 
            Classification.EXCELLENT, 
            Classification.GOOD, 
            Classification.MISTAKE, 
            Classification.BLUNDER
        ];

        displayedClassifications.forEach(classif => {
            const whiteCount = analysis.white.counts[classif.type] || 0;
            const blackCount = analysis.black.counts[classif.type] || 0;
            
            const label = classif.type.charAt(0).toUpperCase() + classif.type.slice(1);
            const row = $(`<div class="stats-row">
                <div class="stats-label">${label}</div>
                <div class="stats-count ${classif.class}">${whiteCount}</div>
                <div class="stats-icon"></div>
                <div class="stats-count ${classif.class}">${blackCount}</div>
            </div>`);
            
            // Add classification icon if available
            if (classif.cachedImg) {
                row.find('.stats-icon').append(classif.cachedImg.cloneNode(true));
            } else {
                // Manually add the icon
                const icon = $(`<img src="${classif.src}" alt="${classif.type}" class="stats-icon">`);
                row.find('.stats-icon').append(icon);
            }
            
            movesSection.append(row);
        });
        
        return movesSection;
    }

    /**
     * Updates an existing game stats display
     * @param {jQuery|string} container - The container element or selector
     * @param {Object} analysis - The updated game analysis data
     * @param {string} whiteName - White player name
     * @param {string} blackName - Black player name
     */
    static update(container, analysis, whiteName = 'White', blackName = 'Black') {
        this.render(container, analysis, whiteName, blackName);
    }
} 