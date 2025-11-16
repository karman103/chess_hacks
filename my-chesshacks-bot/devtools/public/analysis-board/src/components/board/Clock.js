/**
 * Clock utility class for displaying chess clocks from PGN comment data
 */
export class Clock {
    /**
     * Updates the clock display for both players
     * @param {Object} node - The current move tree node
     * @param {boolean} isFlipped - Whether the board is flipped
     */
    static updateClocks(node, isFlipped = false) {
        // Get clock data for current position
        const whiteTime = Clock.getClockTime(node, 'white');
        const blackTime = Clock.getClockTime(node, 'black');
        
        // Update clock displays
        Clock.displayClock('#white-clock .clock-time', whiteTime);
        Clock.displayClock('#black-clock .clock-time', blackTime);
        
        // Handle flipped board - clocks should stay with their respective players
        // The CSS handles the visual positioning
    }
    
    /**
     * Gets the clock time for a specific color from the node
     * @param {Object} node - The move tree node
     * @param {string} color - 'white' or 'black'
     * @returns {number|null} The clock time in milliseconds or null if not available
     */
    static getClockTime(node, color) {
        // Check if this node has clock data from comments
        if (color === 'white' && node.whiteTimeLeft !== undefined) {
            return node.whiteTimeLeft;
        } else if (color === 'black' && node.blackTimeLeft !== undefined) {
            return node.blackTimeLeft;
        }
        
        return null;
    }
    
    /**
     * Displays the clock time in the specified element
     * @param {string} selector - CSS selector for the clock element
     * @param {number|null} milliseconds - The time in milliseconds to display
     * @param {boolean} isActive - Whether this is the active player's clock
     */
    static displayClock(selector, milliseconds, isActive = false) {
        const element = document.querySelector(selector);
        if (!element) return;
        
        // Remove all clock-related classes
        element.classList.remove('low-time', 'active');
        
        if (milliseconds !== null && milliseconds !== undefined) {
            const formattedTime = Clock.formatClockTime(milliseconds);
            element.textContent = formattedTime;
            
            // Add low-time class if time is less than 10 seconds
            if (milliseconds < 10000) {
                element.classList.add('low-time');
            }
        } else {
            element.textContent = '--:--';
        }
        
        // Add active class if this is the active player's clock
        if (isActive) {
            element.classList.add('active');
        }
    }
    
    /**
     * Formats a clock time in milliseconds for display
     * @param {number} milliseconds - Time in milliseconds
     * @returns {string} Formatted time string (e.g., "3:00", "1:30:00")
     */
    static formatClockTime(milliseconds) {
        const totalSeconds = Math.floor(milliseconds / 1000);
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;
        
        if (hours > 0) {
            return `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        } else {
            return `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
    }
    
    /**
     * Resets clock display to default state
     */
    static resetClocks() {
        Clock.displayClock('#white-clock .clock-time', null, false);
        Clock.displayClock('#black-clock .clock-time', null, false);
    }
    
    /**
     * Updates clocks based on the current move tree position
     * @param {Object} moveTree - The move tree instance
     * @param {boolean} isFlipped - Whether the board is flipped
     * @param {string} pgn - Optional PGN string for TimeControl fallback
     */
    static updateFromMoveTree(moveTree, isFlipped = false, pgn = null) {
        if (!moveTree || !moveTree.currentNode) {
            Clock.resetClocks();
            return;
        }
        
        const currentNode = moveTree.currentNode;
        
        // Get initial time from TimeControl or first move
        const getInitialTime = () => {
            // First try to get from TimeControl
            if (pgn) {
                const timeControlTime = Clock.getTimeControlFromPGN(pgn);
                if (timeControlTime) return timeControlTime;
            }
            
            // Fallback to first move's clock data
            if (moveTree.mainline.length > 1) {
                const firstMove = moveTree.mainline[1];
                // Use whichever time is available from the first move
                if (firstMove.whiteTimeLeft !== undefined) {
                    return firstMove.whiteTimeLeft;
                }
                if (firstMove.blackTimeLeft !== undefined) {
                    return firstMove.blackTimeLeft;
                }
            }
            return null;
        };
        
        // For the starting position, show initial time for both players
        if (currentNode.id === 'root') {
            const initialTime = getInitialTime();
            Clock.displayClock('#white-clock .clock-time', initialTime, true); // White to move initially
            Clock.displayClock('#black-clock .clock-time', initialTime, false);
            return;
        }
        
        // Find the most recent clock times for both players
        let whiteTime = null;
        let blackTime = null;
        let nextToMove = 'w'; // Default to white
        
        // Walk backwards through the mainline to find the most recent times
        const nodeIndex = moveTree.getNodeIndex(currentNode);
        if (nodeIndex !== -1) {
            // Determine whose turn it is based on current position
            if (currentNode.move) {
                nextToMove = currentNode.move.color === 'w' ? 'b' : 'w';
            }
            
            // Check current and previous moves for clock data
            for (let i = nodeIndex; i >= 1; i--) {
                const node = moveTree.mainline[i];
                
                // Get white's time if we don't have it yet
                if (!whiteTime && node.whiteTimeLeft !== undefined) {
                    whiteTime = node.whiteTimeLeft;
                }
                
                // Get black's time if we don't have it yet
                if (!blackTime && node.blackTimeLeft !== undefined) {
                    blackTime = node.blackTimeLeft;
                }
                
                // Break if we have both times
                if (whiteTime !== null && blackTime !== null) break;
            }
        }
        
        // If we don't have clock data, use initial time
        const initialTime = getInitialTime();
        if (whiteTime === null) whiteTime = initialTime;
        if (blackTime === null) blackTime = initialTime;
        
        // Update displays with active clock highlighting
        const isWhiteActive = nextToMove === 'w';
        Clock.displayClock('#white-clock .clock-time', whiteTime, isWhiteActive);
        Clock.displayClock('#black-clock .clock-time', blackTime, !isWhiteActive);
    }
    
    /**
     * Sets both clocks to initial starting time before analysis begins
     * @param {Object} moveTree - The move tree instance
     * @param {string} pgn - The PGN string to extract TimeControl from
     */
    static setInitialClocks(moveTree, pgn) {
        if (!moveTree) {
            Clock.resetClocks();
            return;
        }
        
        // Get initial time from PGN TimeControl header
        const initialTime = Clock.getTimeControlFromPGN(pgn);
        
        // Set both clocks to initial time, with white active
        Clock.displayClock('#white-clock .clock-time', initialTime, true);
        Clock.displayClock('#black-clock .clock-time', initialTime, false);
    }
    
    /**
     * Extracts TimeControl value from PGN header and converts to milliseconds
     * @param {string} pgn - The PGN string
     * @returns {number|null} Time in milliseconds or null if not found
     */
    static getTimeControlFromPGN(pgn) {
        if (!pgn) return null;
        
        // Extract TimeControl header value
        const timeControlMatch = pgn.match(/\[TimeControl\s+"([^"]+)"\]/i);
        if (!timeControlMatch) return null;
        
        const timeControlValue = timeControlMatch[1];
        
        // Handle different TimeControl formats
        if (timeControlValue === '-') {
            // Unlimited time
            return null;
        }
        
        // Handle formats like "600" (seconds), "600+5" (seconds + increment), "1800+30"
        const parts = timeControlValue.split('+');
        const baseTimeSeconds = parseInt(parts[0]);
        
        if (isNaN(baseTimeSeconds)) return null;
        
        // Convert seconds to milliseconds
        return baseTimeSeconds * 1000;
    }
}
