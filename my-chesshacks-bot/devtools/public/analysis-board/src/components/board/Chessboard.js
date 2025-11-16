/**
 * chessboard.js - A flexible, customizable chessboard component!
 *
 * @license
 * Copyright (c) 2025, Cooper Ross
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

import { Classification, ClasifCss } from '../../classification/MoveClassifier.js';

export const WHITE = 'w';
export const BLACK = 'b';

export const PAWN = 'p';
export const KNIGHT = 'n';
export const BISHOP = 'b';
export const ROOK = 'r';
export const QUEEN = 'q';
export const KING = 'k';

export const Sound = {
	MOVE: 'move',
	CAPTURE: 'capture',
	CHECK: 'check',
	CASTLE: 'castle',
	PROMOTE: 'promote'
};

export const Css = {
	HIGHLIGHT: 'highlight',
	SELECTED: 'selected-square',
	DROPPABLE: 'ui-droppable-active',
	JUST_MOVED: 'just-moved',
	DROPPABLE_HOVER: 'ui-droppable-hover'
};

export class DOMUtils {
	static $(selector, context = document) {
		return typeof selector === 'string' 
			? selector.startsWith('#') ? context.getElementById(selector.slice(1)) : context.querySelector(selector)
			: selector;
	}

	static $$(selector, context = document) {
		return Array.from(context.querySelectorAll(selector));
	}

	static addClass(element, className) {
		if (!element) return;
		const elements = Array.isArray(element) ? element : [element];
		elements.forEach(el => el.classList.add(...className.split(' ')));
	}

	static removeClass(element, className) {
		if (!element) return;
		const elements = Array.isArray(element) ? element : [element];
		elements.forEach(el => el.classList.remove(...className.split(' ')));
	}

	static toggleClass(element, className) {
		element?.classList.toggle(className);
	}

	static hasClass(element, className) {
		return element?.classList.contains(className);
	}

	static setStyle(element, styles) {
		if (element && typeof styles === 'object') Object.assign(element.style, styles);
	}

	static getOffset(element) {
		const rect = element.getBoundingClientRect();
		return {
			top: rect.top + window.scrollY,
			left: rect.left + window.scrollX,
			width: rect.width,
			height: rect.height
		};
	}

	static createElement(tag, options = {}) {
		const element = document.createElement(tag);
		Object.entries(options).forEach(([key, value]) => {
			if (key === 'attributes') Object.entries(value).forEach(([k, v]) => element.setAttribute(k, v));
			else if (key === 'styles') Object.assign(element.style, value);
			else element[key] = value;
		});
		return element;
	}

	static on(element, events, handler, options = {}) {
		if (element) events.split(' ').forEach(event => element.addEventListener(event, handler, options));
	}

	static off(element, events, handler) {
		if (element) events.split(' ').forEach(event => element.removeEventListener(event, handler));
	}

	static empty(element) {
		if (element) element.innerHTML = '';
	}

	static append(parent, child) {
		if (!parent) return;
		typeof child === 'string' ? parent.insertAdjacentHTML('beforeend', child) : parent.appendChild(child);
	}

	static remove(element) {
		element?.parentNode?.removeChild(element);
	}

	static find(element, selector) {
		return element?.querySelector(selector) ?? null;
	}
	
	static findAll(element, selector) {
		return element?.querySelectorAll ? Array.from(element.querySelectorAll(selector)) : [];
	}

	static createStyleSheet(id) {
		document.getElementById(id)?.remove();
		const style = document.createElement('style');
		style.id = id;
		document.head.appendChild(style);
		return style.sheet;
	}

	static addCSSRule(sheet, selector, rules) {
		const ruleText = Object.entries(rules).map(([prop, val]) => `${prop}: ${val}`).join('; ');
		sheet.insertRule(`${selector} { ${ruleText} }`, sheet.cssRules.length);
	}

	static injectKeyframes(sheet, name, keyframes) {
		const keyframeText = Object.entries(keyframes)
			.map(([key, rules]) => `${key} { ${Object.entries(rules).map(([prop, val]) => `${prop}: ${val}`).join('; ')} }`)
			.join(' ');
		sheet.insertRule(`@keyframes ${name} { ${keyframeText} }`, sheet.cssRules.length);
	}
}

export class Chessboard {
	/**
	 * Creates a new chessboard instance on the target element selector.
	 * @param {string} selector The selector where the chessboard will be instantiated.
	 * @param {Object} settings Configuration settings for the chessboard.
	 * @param {Object} handler Chess handler for move validation and game logic.
	 */
	constructor(selector, settings = {}, handler) {
		// Core properties
		this.selector = selector;
		this.id = Date.now().toString(36);
		this.flipped = false;
		this.selectedPiece = undefined;
		this.pendingPromotion = null;
		this.squareSize = 0;

		// Caches and resources
		this.pieceCache = {};
		this.audioContext = new AudioContext();
		this.volumeNode = this.audioContext.createGain();
		this.audioBuffers = {};
		this.eventListeners = [];
		this.canvasContext = null;
		this.canvas = null;
		this.isDestroyed = false;
		this.styleSheet = null;

		// Game state
		this.arrows = [];
		this.highlights = [];
		this.squares = [];
		this.events = {};
		this.chess = handler;

		// Bound methods
		this.boundOnResize = this._onResize.bind(this);
		this.boundContextMenu = (event) => { event.preventDefault(); return false; };
		this.boundDocumentMouseMove = this._onDocumentMouseMove.bind(this);
		this.boundDocumentMouseUp = this._onDocumentMouseUp.bind(this);
		this.boundOnMouseDown = this._onMouseDown.bind(this);
		this.boundOnMouseMove = this._onMouseMove.bind(this);
		this.boundOnMouseUp = this._onMouseUp.bind(this);

		this.settings = {
			theme: {
				boardLightSquareColor: 'rgba(224, 224, 224, 1)',
				boardDarkSquareColor: 'rgba(110, 161, 118, 1)',
			boardBackgroundPath: '',
			boardImageBackground: false,
			pieceFoldersPath: '/pieces/',
			pieceFolderName: 'cburnett',
			pieceFormat: 'lichess',
			soundFoldersPath: '/sounds/',
			soundFolderName: 'default'
			},
			styling: {
				fontFamily: 'Arial, sans-serif',
				selectedSquareColor: 'rgba(255, 233, 38, 0.27)',
				justMovedColor: 'rgba(255, 208, 0, 0.36)',
				highlightColor: 'rgba(255, 82, 82, 0.71)',
				arrowColor: 'rgba(223, 145, 0, 0.59)',
				droppableIndicatorColor: 'rgba(0, 0, 0, 0.15)',
				droppableHoverBorderColor: 'rgba(255, 255, 255, 0.781)',
				droppableHoverBorderWidth: '5px',
				captureIndicatorSize: 'calc(1.1vmin)',
				captureIndicatorColor: 'rgba(0, 0, 0, 0.15)',
				draggedPieceScale: 1.05,
				hoverCursor: 'grab',
				grabCursor: 'grabbing',
				promotionPanelBackground: 'rgba(61, 61, 61, 0.75)',
				promotionPanelShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',
				promotionPieceHoverColor: 'rgba(0, 0, 0, 0.1)',
				promotionCancelBackground: 'rgb(29, 29, 29)',
				promotionCancelHoverBackground: 'rgb(44, 44, 44)',
				promotionCancelHeight: '40px',
				classificationSize: '50%',
				classificationOffsetX: '150%',
				classificationOffsetY: '-50%',
				classificationBorderOffsetX: '110%',
				classificationBorderOffsetY: '-10%',
				notationFontSize: 'calc(2.5vmin)',
				notationFontWeight: '600',
				notationOffset: '5px'
			},
			callbacks: {
				onMove: null, onMoveCancelled: null, onCheck: null, onCheckmate: null,
				onDraw: null, onStalemate: null, onDragStart: null, onDragMove: null,
				onDrop: null, onPieceClick: null, onPromotion: null, onPromotionStart: null,
				onPromotionComplete: null, onPositionChange: null, onOrientationChange: null,
				onClear: null, onDestroy: null, onHighlight: null, onArrowCreate: null, 
				onUserMove: null,
			},

			audioEnabled: true,
			isInteractive: true,
			showBoardLabels: true,

			draggingEnabled: true,
			clickingEnabled: true,

			pieceDragThreshold: -1,
			pieceClickThreshold: 40,

			pieceAnimationDuration: 0.1,
			pieceAnimationEasing: 'ease-out',
			
			pieceRevertDuration: 0.1,
			pieceRevertEasing: 'ease-out'
		};

		if (settings && typeof settings === 'object') this._mergeSettings(settings);
		this._initializeCallbacks();
		this.init();
	}

	/**
	 * Initializes the chessboard UI inside a container: board layout, input handlers, and audio/piece caching.
	 * @private
	 */
	init() {
		if (this.isDestroyed) return;
		
		this._createStyleSheet();
		const container = DOMUtils.$(this.selector);

		DOMUtils.append(container, `
			<div id="${this.id}" class='chessboard'>
				<canvas id='overlay-${this.id}' data-chessboard-id='${this.id}' class='board-overlay'></canvas>
				<div id='squares-${this.id}' data-chessboard-id='${this.id}' class='board-squares'></div>
			</div>
		`);

		const chessboard = DOMUtils.$('#squares-' + this.id);
		for (let i = 0; i < 64; i++) {
			const color = (Math.floor(i / 8) + i) % 2 ? 'dark' : 'light';
			const square = DOMUtils.createElement('div', {
				className: `square ${color}`,
				attributes: { 'data-square': i.toString() }
			});
			DOMUtils.append(chessboard, square);
			this.squares.push(square);
		}

		if (!this.settings.isInteractive) DOMUtils.setStyle(container, { 'pointer-events': 'none' });

		this._cachePieces();
		this._cacheAudio();
		this._addTrackedEventListener(window, 'resize', this.boundOnResize);
		this._addTrackedEventListener(container, 'contextmenu', this.boundContextMenu);
		this.fen();
		this._initializeInput();
		this._render();
	}

	/**
	 * Destroys the chessboard instance and cleans up all resources to prevent memory leaks.
	 * This method should be called when the chessboard is no longer needed.
	 * 
	 * @example
	 * // Clean up the board when component unmounts or page navigation occurs
	 * board.destroy();
	 */
	destroy() {
		if (this.isDestroyed) return;
		
		// Emit destroy event before cleanup
		this.emit('destroy');
		
		// Mark as destroyed to prevent further operations
		this.isDestroyed = true;
		
		// Clean up the dynamic stylesheet
		if (this.styleSheet) {
			const styleElement = document.getElementById(`chessboard-styles-${this.id}`);
			if (styleElement) {
				styleElement.remove();
			}
			this.styleSheet = null;
		}
		
		// Clean up all tracked event listeners
		this.eventListeners.forEach(({ element, eventName, handler }) => {
			try {
				element.removeEventListener(eventName, handler);
			} catch (error) {
				console.warn('Error removing event listener:', error);
			}
		});
		this.eventListeners = [];
		
		// Clean up AudioContext and related resources
		if (this.audioContext) {
			try {
				// Disconnect all audio nodes
				if (this.volumeNode) {
					this.volumeNode.disconnect();
					this.volumeNode = null;
				}
				
				// Close the audio context
				if (this.audioContext.state !== 'closed') {
					this.audioContext.close();
				}
				this.audioContext = null;
			} catch (error) {
				console.warn('Error cleaning up AudioContext:', error);
			}
		}
		
		// Clear audio buffers
		this.audioBuffers = {};
		
		// Clear piece cache and other caches
		this.pieceCache = {};
		
		// Clear canvas context reference
		this.canvasContext = null;
		this.canvas = null;
		
		// Clear game state
		this.selectedPiece = undefined;
		this.pendingPromotion = null;
		this.dragPiece = undefined;
		this.hoveredSquare = undefined;
		this.dragStartXY = null;
		this.startDragIndex = null;

		// Clear board state
		this.arrows = [];
		this.highlights = [];
		this.squares = [];
		
		// Remove DOM elements
		try {
			const boardElement = DOMUtils.$(`#${this.id}`);
			if (boardElement) {
				DOMUtils.remove(boardElement);
			}
		} catch (error) {
			console.warn('Error removing DOM elements:', error);
		}
		
		// Clear references to bound methods
		this.boundOnResize = null;
		this.boundContextMenu = null;
		this.boundDocumentMouseMove = null;
		this.boundDocumentMouseUp = null;
		this.boundOnMouseDown = null;
		this.boundOnMouseMove = null;
		this.boundOnMouseUp = null;
		
		// Clear chess instance
		this.chess = null;	
	}

	/**
	 * Helper method to check if board is destroyed or not interactive
	 * @private
	 */
	_canInteract() {
		return !this.isDestroyed && this.settings.isInteractive;
	}

	/**
	 * Sets a single option or multiple options at once.
	 * @param {string|Object} keyOrOptions - The key of the option or an object of options.
	 * @param {*} value - The value to set for the option.
	 * @returns {Promise} A promise that resolves when the options are set.
	 */
	async setOption(keyOrOptions, value) {
		const updates = typeof keyOrOptions === 'string' 
			? { [keyOrOptions]: value }
			: this._flattenOptions(keyOrOptions);
		
		let stylingChanged = false;
		let piecesChanged = false;
		let audioChanged = false;
		
		for (const [key, val] of Object.entries(updates)) {
			this._setNestedOption(key, val);
			
			// Check for styling changes
			if (key.startsWith('styling.') || key.startsWith('theme.board') || key === 'showBoardLabels') {
				stylingChanged = true;
			}
			
			// Check for piece-related changes
			if (key.startsWith('theme.piece') || key.startsWith('theme.customPieceUrls')) {
				piecesChanged = true;
			}
			
			// Check for audio-related changes
			if (key.startsWith('theme.sound') || key.startsWith('theme.customSoundUrls') || key === 'audioEnabled') {
				audioChanged = true;
			}
		}
		
		// Only clear and reload caches that actually need updating
		if (piecesChanged) {
			this.pieceCache = {};
			await this._cachePieces();
		}
		
		if (audioChanged) {
			this.audioBuffers = {};
			this._cacheAudio();
		}
		
		if (stylingChanged && this.styleSheet) {
			this._createStyleSheet();
		}

		this.refresh(true);
	}

	/**
	 * Gets the value of an option using dot notation
	 * 
	 * @param {string} path - The option path (e.g., 'theme.boardBackgroundPath')
	 * @returns {*} The option value
	 * 
	 * @example
	 * const isInteractive = board.getOption('isInteractive');
	 * const arrowColor = board.getOption('styling.arrowColor');
	 */
	getOption(path) {
		const keys = path.split('.');
		let current = this.settings;
		for (const key of keys) {
			if (!(key in current)) throw new Error(`Invalid option: ${path}`);
			current = current[key];
		}
		return current;
	}

	/**
	 * Merges user settings with default settings recursively
	 * @private
	 * @param {Object} userSettings - User provided settings
	 */
	_mergeSettings(userSettings) {
		const merge = (target, source) => {
			for (const key in source) {
				if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
					if (!target[key] || typeof target[key] !== 'object') target[key] = {};
					merge(target[key], source[key]);
				} else {
					target[key] = source[key];
				}
			}
		};
		merge(this.settings, userSettings);
	}

	/**
	 * Internal helper to set nested options using dot notation
	 * @private
	 * @param {string} path - The property path (e.g., 'theme.boardBackgroundPath')
	 * @param {*} value - The value to set
	 */
	_setNestedOption(path, value) {
		const parts = path.split('.');
		const lastPart = parts.pop();
		const target = parts.reduce((obj, key) => obj[key] = obj[key] || {}, this.settings);
		target[lastPart] = value;
	}

	/**
	 * Internal helper to flatten nested objects into dot notation paths
	 * @private
	 * @param {Object} obj - The object to flatten
	 * @param {string} [prefix=''] - The current path prefix
	 * @returns {Object} Flattened object with dot notation keys
	 */
	_flattenOptions(obj, prefix = '') {
		return Object.entries(obj).reduce((acc, [key, value]) => {
			const newKey = prefix ? `${prefix}.${key}` : key;
			return value && typeof value === 'object' && !Array.isArray(value)
				? { ...acc, ...this._flattenOptions(value, newKey) }
				: { ...acc, [newKey]: value };
		}, {});
	}

	/**
	 * Sets a custom URL for a specific piece.
	 * 
	 * @param {string} color - The piece color ('w' for white, 'b' for black)
	 * @param {string} type - The piece type ('k', 'q', 'r', 'b', 'n', 'p')
	 * @param {string} url - The URL to the piece image
	 * 
	 * @example
	 * // Set a custom white king image
	 * board.setCustomPieceUrl('w', 'k', 'https://example.com/white_king.svg');
	 * 
	 * @example
	 * // Set a custom black queen image
	 * board.setCustomPieceUrl('b', 'q', '/assets/custom/black_queen.svg');
	 */
	async setCustomPieceUrl(color, type, url) {
		if (!['w', 'b'].includes(color)) throw new Error("Color must be 'w' or 'b'");
		if (!['k', 'q', 'r', 'b', 'n', 'p'].includes(type)) throw new Error("Type must be one of: 'k', 'q', 'r', 'b', 'n', 'p'");
		
		this.settings.theme.customPieceUrls[`${color}_${type}`] = url;
		this.pieceCache = {};
		await this._cachePieces();
		this.refresh(true);
	}

	/**
	 * Sets a custom URL for a specific sound effect.
	 * 
	 * @param {string} sound - The sound type ('move', 'capture', 'check', 'castle', 'promote')
	 * @param {string} url - The URL to the audio file
	 * 
	 * @example
	 * // Set a custom move sound
	 * board.setCustomSoundUrl('move', 'https://example.com/move.mp3');
	 * 
	 * @example
	 * // Set a custom capture sound
	 * board.setCustomSoundUrl('capture', '/sounds/custom/custom_capture.wav');
	 */
	async setCustomSoundUrl(sound, url) {
		if (!Object.values(Sound).includes(sound)) throw new Error(`Invalid sound: ${sound}`);
		this.settings.theme.customSoundUrls[sound] = url;
		this.audioBuffers = {};
		await this._cacheAudio();
	}

	/**
	 * Removes a custom piece URL, reverting to the default folder-based approach.
	 * 
	 * @param {string} color - The piece color ('w' for white, 'b' for black)
	 * @param {string} type - The piece type ('k', 'q', 'r', 'b', 'n', 'p')
	 * 
	 * @example
	 * // Remove custom white king, use default
	 * board.removeCustomPieceUrl('w', 'k');
	 */
	async removeCustomPieceUrl(color, type) {
		delete this.settings.theme.customPieceUrls[`${color}_${type}`];
		this.pieceCache = {};
		await this._cachePieces();
		this.refresh(true);
	}

	/**
	 * Removes a custom sound URL, reverting to the default folder-based approach.
	 * 
	 * @param {string} sound - The sound type ('move', 'capture', 'check', 'castle', 'promote')
	 * 
	 * @example
	 * // Remove custom move sound, use default
	 * board.removeCustomSoundUrl('move');
	 */
	async removeCustomSoundUrl(sound) {
		delete this.settings.theme.customSoundUrls[sound];
		this.audioBuffers = {};
		await this._cacheAudio();
	}

	/**
	 * Add an event listener
	 * @param {string} event - Event name
	 * @param {Function} callback - Callback function
	 */
	on(event, callback) {
		if (!this.events[event]) this.events[event] = [];
		this.events[event].push(callback);
		return this;
	}

	/**
	 * Remove an event listener
	 * @param {string} event - Event name
	 * @param {Function} callback - Callback function to remove
	 */
	off(event, callback) {
		if (!this.events[event]) return this;
		if (callback) {
			this.events[event] = this.events[event].filter(cb => cb !== callback);
		} else {
			delete this.events[event];
		}
		return this;
	}

	/**
	 * Add a one-time event listener
	 * @param {string} event - Event name
	 * @param {Function} callback - Callback function
	 */
	once(event, callback) {
		const onceWrapper = (...args) => {
			callback(...args);
			this.off(event, onceWrapper);
		};
		return this.on(event, onceWrapper);
	}

	/**
	 * Emit an event
	 * @param {string} event - Event name
	 * @param {...any} args - Arguments to pass to callbacks
	 * @returns {*} Return value from callbacks (for prevention)
	 */
	emit(event, ...args) {
		if (!this.events[event]) return this;
		let result = undefined;
		this.events[event].forEach(callback => {
			try {
				const callbackResult = callback(...args);
				if (callbackResult !== undefined) result = callbackResult;
			} catch (error) {
				console.error(`Error in event handler for '${event}':`, error);
			}
		});
		return result !== undefined ? result : this;
	}

	/**
	 * Remove all event listeners
	 */
	removeAllListeners() {
		this.events = {};
		return this;
	}

	/**
	 * Initializes callback-based event listeners from settings
	 * @private
	 */
	_initializeCallbacks() {
		const callbacks = this.settings.callbacks;
		if (!callbacks) return;
		Object.entries(callbacks).forEach(([eventName, callback]) => {
			if (typeof callback === 'function') {
				const event = eventName.replace(/^on/, '').toLowerCase();
				this.on(event, callback);
			}
		});
	}

	/**
	 * Helper method to track event listeners
	 * @private
	 * @param {HTMLElement|Window|Document} element - The element to add event listener to
	 * @param {string} events - Space-separated event names
	 * @param {Function} handler - The event handler function
	 * @param {Object} options - Event listener options
	 */
	_addTrackedEventListener(element, events, handler, options = {}) {
		if (this.isDestroyed) return;
		events.split(' ').forEach(eventName => {
			element.addEventListener(eventName, handler, options);
			this.eventListeners.push({ element, eventName, handler, options });
		});
	}

	/**
	 * Helper method to remove specific tracked event listeners
	 * @private
	 * @param {HTMLElement|Window|Document} element - The element to remove event listener from
	 * @param {string} events - Space-separated event names
	 * @param {Function} handler - The event handler function
	 */
	_removeTrackedEventListener(element, events, handler) {
		events.split(' ').forEach(eventName => {
			element.removeEventListener(eventName, handler);
			this.eventListeners = this.eventListeners.filter(listener => 
				!(listener.element === element && listener.eventName === eventName && listener.handler === handler)
			);
		});
	}

	/**
	 * Gets the complete board state
	 * @returns {Object} Board state object
	 */
	getState() {
		return {
			fen: this.chess._currentFen || 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
			flipped: this.flipped,
			selectedPiece: this.selectedPiece,
			arrows: [...this.arrows],
			highlights: [...this.highlights],
			history: this.chess.history ? this.chess.history({ verbose: true }) : [],
			turn: this.chess.turn(),
		};
	}

	/**
	 * Restores a saved board state
	 * @param {Object} state - Previously saved state object
	 */
	setState(state) {
		if (state.fen) {
			this.fen(state.fen);
		}
		if (typeof state.flipped === 'boolean' && state.flipped !== this.flipped) {
			this.flip();
		}
		if (state.arrows) {
			this.arrows = [...state.arrows];
		}
		if (state.highlights) {
			this.highlights = [...state.highlights];
			// Reapply highlights to DOM
			this.highlights.forEach(highlight => {
				const index = this.flipped ? 63 - highlight : highlight;
				const square = this.getSquare(index);
				square.classList.add(Css.HIGHLIGHT);
			});
		}
		this._render(); // Redraw arrows
		this.emit('positionchange', state.fen);
	}

	/**
	 * Loads a fen string to the board.
	 * 
	 * @param {string} fen The FEN string to load
	 */
	fen(fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1') {
		this.chess.load(fen);
		this.refresh();
		this.clearBoardElements();
		this.clearBoardHighlights();
	}

	/**
	 * Flips the display board perspective from white to black.
	 */
	flip() {
		this.flipped = !this.flipped;
		this.refresh(true);
		this.clearBoardHighlights();

		this.highlights.forEach(highlight => {
			const index = this.flipped ? 63 - highlight : highlight;
			this.getSquare(index).classList.add(Css.HIGHLIGHT);
		});

		this.arrows.forEach(arrow => {
			arrow.forEach((_, j) => arrow[j] = 63 - arrow[j]);
		});
		
		// Update board labels for the new orientation
		this._updateBoardLabels();
		
		this._render();
		this.emit('orientationchange', this.flipped);
	}

	/**
	 * Refreshes the chessboard UI by clearing highlights and loading the board state.
	 * @param {boolean} clearHighlights Whether to clear highlights (arrows, highlights, move indicators, etc.)
	 */
	refresh(clearHighlights = false) {
		this.selectedPiece = undefined;
		if (!clearHighlights) {
			this.clearBoardElements();
			this.clearBoardHighlights();
		}
		DOMUtils.$$(`#squares-${this.id} .square`).forEach(DOMUtils.empty);
		this._load();
	}

	/**
	 * Gets all legal moves for the current position
	 * @returns {Array} Array of legal moves
	 */
	getPossibleMoves() {
		if (this.chess.moves) {
			return this.chess.moves({ verbose: true });
		}
		return [];
	}

	/**
	 * Gets the move history
	 * @param {Object} options - Options for history format
	 * @returns {Array} Move history
	 */
	getMoveHistory(options = {}) {
		if (this.chess.history) {
			return this.chess.history(options);
		}
		return [];
	}

	/**
	 * Gets detailed information about a square
	 * @param {string} square - Square in algebraic notation (e.g., 'e4')
	 * @returns {Object} Square information
	 */
	getSquareInfo(square) {
		const piece = this.chess.get ? this.chess.get(square) : null;
		const moves = this.chess.moves ? this.chess.moves({ square, verbose: true }) : [];
		
		return {
			square,
			piece,
			legalMoves: moves,
			isAttacked: false, // Would need engine support
			isDefended: false, // Would need engine support
		};
	}

	/**
	 * Exports the current game in PGN format
	 * @returns {string} PGN string
	 */
	exportPGN() {
		if (this.chess.pgn) {
			return this.chess.pgn();
		}
		// Fallback: simple move list
		const history = this.getMoveHistory();
		return history.join(' ');
	}

	/**
	 * Imports a game from PGN format
	 * @param {string} pgn - PGN string
	 * @returns {boolean} Success status
	 */
	importPGN(pgn) {
		try {
			if (this.chess.loadPgn) {
				const success = this.chess.loadPgn(pgn);
				if (success) {
					this.refresh();
					this.emit('positionchange', this.chess._currentFen);
				}
				return success;
			}
			return false;
		} catch (error) {
			console.error('Error importing PGN:', error);
			return false;
		}
	}

	/**
	 * Moves a piece on the chessboard from one square to another.
	 *
	 * @param {Object} move - The move object containing from and to squares.
	 * @param {boolean} animate - Whether to animate the move.
	 * @param {string|object} classification - The move classification to display, either the name or an object.
	 * @param {string} fenBefore - The FEN string before making the move.
	 * @param {boolean} showPromotionUI - Whether to show the promotion UI.
	 * @param {string} promotedPiece - The piece to promote to.
	 * @return {Object|null} The move result or null if promotion in progress.
	 */
	move(move, animate = false, classification = undefined, fenBefore = null, showPromotionUI = true, promotedPiece = null, wasUserMove = false) {
		// Don't allow moves if a promotion is in progress
		if (this.pendingPromotion) return null;

		const fromIdx = this.algebraicToIndex(move.from, this.flipped);
		const toIdx = this.algebraicToIndex(move.to, this.flipped);
		
		const currentSquare = this.getSquare(fromIdx);
		const targetSquare = this.getSquare(toIdx);

		// Store the selected piece
		this.selectedPiece = fromIdx;
		
		// Check if this is a pawn promotion and handle accordingly
		if (showPromotionUI && this._isPawnPromotion(move.from, move.to)) {
			this.pendingPromotion = {
				from: move.from,
				to: move.to,
				fromIdx: fromIdx,
				toIdx: toIdx,
				animate: animate,
				fenBefore: fenBefore
			};
			
			// Emit promotion start event
			this.emit('promotionstart', move.to);
			
			this._showPromotionPanel(fromIdx, toIdx);
			return null;
		} else if (this._isPawnPromotion(move.from, move.to)) {
			this.pendingPromotion = {
				from: move.from,
				to: move.to,
				fromIdx: fromIdx,
				toIdx: toIdx,
				animate: animate,
				fenBefore: fenBefore
			};
			this._completePromotion(promotedPiece, false);
			if (classification) this.addClassification(classification, currentSquare, targetSquare);
			return null;
		}
		
		if (fenBefore) this.chess.load(fenBefore);
		const piece = DOMUtils.$('img', currentSquare);

		if (!currentSquare || !piece) {
			return console.error(`Invalid move: No piece at square ${move.from}`);
		}

		this._updateBoard(currentSquare, targetSquare, piece);

		const moveResult = this.chess.move({
			from: move.from, 
			to: move.to,
			promotion: promotedPiece
		});
		
		if (!moveResult) {
			return console.error(`Illegal move from ${move.from} to ${move.to}`);
		}

		// Always animate the piece movement first
		if (animate) {
			this._animatePiece(piece, currentSquare);
		}

		// Handle special moves after animation
		const handled = this._handleSpecialMoves(moveResult, move.to, classification);
		if (classification) {
			this.addClassification(classification, currentSquare, targetSquare);
		}
		if (!handled) this._playSoundBasedOnOutcome(moveResult);
		
		// Emit events
		this.emit('move', moveResult);
		this.emit('positionchange', this.chess._currentFen);

		if (wasUserMove) this.emit('usermove', moveResult);
		
		// Check for game state changes
		if (this.chess.inCheck && this.chess.inCheck()) {
			this.emit('check', this.chess.turn());
		}
		
		return moveResult;
	}

	/**
	 * Undoes the last move made on the chessboard.
	 * 
	 * @param {boolean} [animate=false] - Whether to animate the move.
	 * @param {Object} [lastMove=null] - The move we want to undo.
	 * @param {string} [fenBefore=null] - The FEN string of the position before this move.
	 * @return {Object|null} - The undone move object, or null if no move to undo.
	 */
	unmove(animate = false, lastMove = null, fenBefore = null) {
		// If a specific move is provided, use it; otherwise get the last move from history
		if (!lastMove) {
			// Get the history from chess.js
			const history = this.chess.history({ verbose: true });
			if (history.length === 0) {
				return null;
			}
			
			// Get the last move
			lastMove = history[history.length - 1];
		}
		
		// Convert algebraic notation to board indices
		const fromIndex = this.algebraicToIndex(lastMove.from, this.flipped);
		const toIndex = this.algebraicToIndex(lastMove.to, this.flipped);
		
		// Special case for promotion - we need to handle it differently
		const isPromotion = !!lastMove.promotion;
		
		// Get the squares
		const fromSquare = this.getSquare(fromIndex);
		const toSquare = this.getSquare(toIndex);
		
		// For promotions, we don't want to move the promoted piece back,
		// since we'll replace it with a pawn at the from square
		if (!isPromotion) {
			const piece = DOMUtils.$('img', toSquare);
			
			if (!toSquare || !piece) {
				console.error(`Invalid unmove: No piece at square ${toIndex}`);
				return null;
			}
			
			// Update the board for regular moves
			this._updateBoard(toSquare, fromSquare, piece, true);
			if (animate) this._animatePiece(piece, toSquare);
		} else {
			// For promotions, we'll handle the animation in handleUndoPromotion
			DOMUtils.empty(toSquare);
		}

		// Handle special moves first
		if (this._handleSpecialUnmoves(lastMove, animate)) {
			// Special move was handled
		} else {
			// Restore captured piece if any
			if (lastMove.captured) {
				const capturedPieceColor = lastMove.color === WHITE ? BLACK : WHITE;
				this._createPiece(toIndex, lastMove.captured, capturedPieceColor);
				this._playSound(Sound.CAPTURE);
			} else {
				this._playSound(Sound.MOVE);
			}
		}
		
		// Use the provided fen or undo the move in the chess.js instance
		if (fenBefore) {
			this.chess.load(fenBefore);
		} else {
			this.chess.undo();
		}
		
		return lastMove;
	}

	/**
	 * Toggles an arrow between two squares on the chessboard.
	 * If an arrow already exists, it is removed; otherwise, it is added.
	 *
	 * @param {HTMLElement} start The starting square element.
	 * @param {HTMLElement} end The ending square element.
	 * @param {Object} options Optional arrow styling options (color, opacity)
	 */
	createArrow(start, end, options = {}) {
		const startIndex = this.getSquareIndex(start);
		const endIndex = this.getSquareIndex(end);

		// Find the existing arrow
		const existingArrow = this.arrows.find(arrow => {
			const [s, e] = Array.isArray(arrow) ? arrow : [arrow.from, arrow.to];
			return s === startIndex && e === endIndex;
		});
		
		// Toggle arrow state
		if (existingArrow) {
			this.arrows = this.arrows.filter(arrow => arrow !== existingArrow);
		} else {
			// If options are provided, store as object; otherwise use legacy array format
			if (options.color || options.opacity !== undefined) {
				this.arrows.push({
					from: startIndex,
					to: endIndex,
					color: options.color || this.settings.styling.arrowColor,
					opacity: options.opacity !== undefined ? options.opacity : 1
				});
			} else {
				this.arrows.push([startIndex, endIndex]);
			}
		}
		
		// Re-render the board
		this._render();
		
		// Emit event
		const fromSquare = this.indexToAlgebraic(startIndex, this.flipped);
		const toSquare = this.indexToAlgebraic(endIndex, this.flipped);
		this.emit('arrowcreate', fromSquare, toSquare);
	}

	/**
	 * Adds an arrow to the board without toggling (always adds, never removes).
	 * @param {HTMLElement} start The starting square element.
	 * @param {HTMLElement} end The ending square element.
	 * @param {Object} options Optional arrow styling options (color, opacity, isProbability)
	 */
	addArrow(start, end, options = {}) {
		const startIndex = this.getSquareIndex(start);
		const endIndex = this.getSquareIndex(end);

		this.arrows.push({
			from: startIndex,
			to: endIndex,
			color: options.color || this.settings.styling.arrowColor,
			opacity: options.opacity !== undefined ? options.opacity : 1,
			isProbability: options.isProbability || false
		});
		
		this._render();
	}

	/**
	 * Clears all probability arrows from the board.
	 */
	clearProbabilityArrows() {
		this.arrows = this.arrows.filter(arrow => {
			// Keep legacy array format arrows (best move arrows) and non-probability arrows
			return Array.isArray(arrow) || !arrow.isProbability;
		});
		this._render();
	}

	/**
	 * Clears highlighted squares and removes all arrows from the board.
	 */
	clearBoardElements() {
		DOMUtils.removeClass(DOMUtils.$$(`#squares-${this.id} .square`), Css.HIGHLIGHT);
		this.arrows = [];
		this.highlights = [];
		this._render();
		this.emit('clear');
	}

	/**
	 * Clears all board classifications and highlights without affecting other elements.
	 */
	clearBoardHighlights() {
		const classifications = DOMUtils.findAll(DOMUtils.$(`#squares-${this.id}`), '.classification');
		for (const classification of classifications) {
			DOMUtils.remove(classification);
		}

		const cssClassifications = DOMUtils.findAll(DOMUtils.$(`#squares-${this.id}`), '.board-classification');
		for (const classification of cssClassifications) {
			DOMUtils.removeClass(classification, 'board-classification');
		}
		
		const allClasses = Object.values(Css);
		for (const className of allClasses) {
			DOMUtils.removeClass(DOMUtils.$$(`#squares-${this.id} .square`), className);
		}

		const allClassifications = Object.values(ClasifCss);
		for (const className of allClassifications) {
			DOMUtils.removeClass(DOMUtils.$$(`#squares-${this.id} .square`), className);
		}
	}

	/**
	 * Attempts to move a piece from one square index to another.
	 * @private
	 */
	_attemptMove(fromIndex, toIndex, animate = false) {
		const legalDestinations = this._getLegalDestinations(fromIndex);
		if (!legalDestinations.includes(toIndex) || this.pendingPromotion !== null) {
			// Emit move cancelled event
			const fromSquare = this.indexToAlgebraic(fromIndex, this.flipped);
			const toSquare = this.indexToAlgebraic(toIndex, this.flipped);
			this.emit('movecancelled', { from: fromSquare, to: toSquare });
			return false;
		}
		
		this.move({
			from: this.indexToAlgebraic(fromIndex, this.flipped),
			to: this.indexToAlgebraic(toIndex, this.flipped)
		}, animate, undefined, undefined, true, undefined, true);
		return true;
	}

	/**
	 * Returns the list of legal destination square indices for a piece at the given board index.
	 * @private
	 */
	_getLegalDestinations(fromIndex) {
		if (this.pendingPromotion) return [];

		const fromAlgebraic = this.indexToAlgebraic(fromIndex, this.flipped);
		const moves = this.chess.moves({ square: fromAlgebraic, verbose: true });
		
		return moves.map(move => this.algebraicToIndex(move.to, this.flipped));
	}

	/**
	 * Handles special moves like castling, promotion, and en passant.
	 * @private
	 * @param {Object} moveResult The result from chess.move().
	 * @param {number} targetSquareIndex The target square index.
	 * @param {string} classification The classification of the move.
	 * @return {boolean} True if a special move was processed; otherwise, false.
	 */
	_handleSpecialMoves(moveResult, targetSquareIndex, classification) {
		if (this._handleCastling(moveResult)) return true;
		if (this._handleEnPassant(moveResult)) return true;
		if (this._handlePromotion(moveResult, targetSquareIndex, classification)) return true;
		
		return false;
	}

	/**
	 * Handles castling moves, specifically the rook movement.
	 * @private
	 * @param {Object} moveResult The move result object.
	 * @return {boolean} True if castling was handled; otherwise, false.
	 */
	_handleCastling(moveResult) {
		const castleType = moveResult.isKingsideCastle() 
			? 'kingside' 
			: moveResult.isQueensideCastle() 
				? 'queenside' 
				: null;
				
		if (!castleType) return false;
		
		const castleMap = {
			w: {kingside: ['h8', 'f8'], queenside: ['a8', 'd8']},
			b: {kingside: ['h1', 'f1'], queenside: ['a1', 'd1']}
		};
		
		const opponentColor = moveResult.color === WHITE ? BLACK : WHITE;
		const [from, to] = castleMap[opponentColor][castleType];
		const fromSquare = this.getSquare(this.algebraicToIndex(from, this.flipped));
		const toSquare = this.getSquare(this.algebraicToIndex(to, this.flipped));
		const piece = DOMUtils.$('img', fromSquare);
		
		this._updateBoard(fromSquare, toSquare, piece, false);
		this._animatePiece(piece, fromSquare);
		this._playSound(Sound.CASTLE);
		
		return true;
	}

	/**
	 * Handles en passant moves.
	 * @private
	 * @param {Object} moveResult The move result object.
	 * @return {boolean} True if en passant was handled; otherwise, false.
	 */
	_handleEnPassant(moveResult) {
		if (moveResult.isEnPassant()) {
			const direction = this.chess.turn() === WHITE ? -1 : 1;
			const offset = this.flipped ? -8 : 8;
			const capturedSquareIndex = 
				this.algebraicToIndex(moveResult.to, this.flipped) + offset * direction;
			const capturedSquare = this.getSquare(capturedSquareIndex);
			if(capturedSquare) DOMUtils.empty(capturedSquare);
			this._playSound(Sound.CAPTURE);
			return true;
		}
		return false;
	}

	/**
	 * Handles undoing special moves like castling, promotion, and en passant.
	 * @private
	 * @param {Object} lastMove - The move details from chess.js history.
	 * @param {boolean} animate - Whether to animate the move.
	 * @return {boolean} True if a special move was processed; otherwise, false.
	 */
	_handleSpecialUnmoves(lastMove, animate = false) {
		if (this._handleUndoCastling(lastMove, animate)) return true;
		if (this._handleUndoPromotion(lastMove, animate)) return true;
		if (this._handleUndoEnPassant(lastMove)) return true;
		
		return false;
	}

	/**
	 * Handles undoing castling moves, specifically reversing the rook movement.
	 * @private
	 * @param {Object} moveResult The move result object.
	 * @param {boolean} animate Whether to animate the move.
	 * @return {boolean} True if castling undo was handled; otherwise, false.
	 */
	_handleUndoCastling(moveResult, animate = false) {
		const castleType = moveResult.isKingsideCastle() 
			? 'kingside' 
			: moveResult.isQueensideCastle() 
				? 'queenside' 
				: null;
				
		if (!castleType) return false;

		// For undoing, the rook needs to go from its castled position back to its original position
		const castleMap = {
			w: {kingside: ['f8', 'h8'], queenside: ['d8', 'a8']},
			b: {kingside: ['f1', 'h1'], queenside: ['d1', 'a1']}
		};
		
		const opponentColor = moveResult.color === WHITE ? BLACK : WHITE;
		const [from, to] = castleMap[opponentColor][castleType];
		const fromSquare = this.getSquare(this.algebraicToIndex(from, this.flipped));
		const toSquare = this.getSquare(this.algebraicToIndex(to, this.flipped));
		const piece = DOMUtils.$('img', fromSquare);
		
		this._updateBoard(fromSquare, toSquare, piece, false);
		if (animate) this._animatePiece(piece, fromSquare);
		this._playSound(Sound.CASTLE);
		
		return true;
	}

	/**
	 * Handles undoing en passant moves.
	 * @private
	 * @param {Object} moveResult - The move result object.
	 * @return {boolean} True if en passant undo was handled; otherwise, false.
	 */
	_handleUndoEnPassant(moveResult) {
		if (moveResult.isEnPassant()) {
			const direction = this.chess.turn() === WHITE ? -1 : 1;
			const offset = this.flipped ? -8 : 8;
			const capturedSquare = this.algebraicToIndex(moveResult.to, this.flipped) + offset * direction;

			this._createPiece(capturedSquare, moveResult.captured, this.chess.turn());
			this._playSound(Sound.CAPTURE);
			return true;
		}
		return false;
	}

	/**
	 * Handles pawn promotion.
	 * @private
	 * @param {Object} moveResult The move result object.
	 * @param {number} targetSquareIndex The target square index.
	 * @param {string} classification The classification of the move.
	 * @return {boolean} True if promotion was handled; otherwise, false.
	 */
	_handlePromotion(moveResult, targetSquareIndex, classification) {
		if (moveResult.isPromotion()) {
			// For auto-promotion (non-interactive) or pre-selected promotion
			const color = this.chess.turn() === WHITE ? BLACK : WHITE;
			const promoted = moveResult.promotion || QUEEN;

			// Just for auto-promotion (when not using the panel)
			if (!this.pendingPromotion) {
				const targetSquare = this.getSquare(targetSquareIndex);
				// Is this bad? Yes. Does it work for now? Yes.
				setTimeout(() => {
					DOMUtils.empty(targetSquare);
					this._createPiece(targetSquareIndex, promoted, color);
					this._playSound(Sound.PROMOTE);

					// Since we emptied the square, the classification will be lost
					if (classification) {
						this.addClassification(classification, targetSquare, targetSquare);
					}
				}, 50);
			}
			
			return true;
		}
		return false;
	}

	/**
	 * Checks if a move is a pawn promotion.
	 * @private
	 * @param {string} fromAlg The algebraic notation of the source square.
	 * @param {string} toAlg The algebraic notation of the target square.
	 * @return {boolean} Whether the move is a pawn promotion.
	 */
	_isPawnPromotion(fromAlg, toAlg) {
		const piece = this.chess.get(fromAlg);
		
		// If no piece or not a pawn, return false
		if (!piece || piece.type !== PAWN) return false;
		
		// Check if the pawn is moving to the last rank
		const toRank = parseInt(toAlg[1]);
		return (piece.color === WHITE && toRank === 8) || 
				(piece.color === BLACK && toRank === 1);
	}

	/**
	 * Shows the promotion panel for selecting a promotion piece.
	 * @private
	 * @param {number} fromSquareId The index of the square the pawn is moving from.
	 * @param {number} toSquareId The index of the square the pawn is promoting on.
	 */
	_showPromotionPanel(fromSquareId, toSquareId) {
		const board = this;
		DOMUtils.remove(DOMUtils.$(`#promotion-panel-${this.id}`));

		// Clear any existing highlights and legal move indicators
		this.clearBoardElements();
		DOMUtils.removeClass(DOMUtils.$$(`#squares-${this.id} .square`), Css.DROPPABLE);

		const target = this.getSquare(toSquareId);
		const color = this.chess.turn();
		const bottom = Math.floor(toSquareId / 8) > 4;

		const panel = DOMUtils.createElement('div', { id: `promotion-panel-${this.id}`, className: 'promotion-panel' });
		DOMUtils.append(target, panel);
		
		if (bottom) {
			DOMUtils.addClass(panel, 'bottom');
			DOMUtils.setStyle(panel, { bottom: '0px' });
		} else {
			DOMUtils.setStyle(panel, { top: '0px' });
		}

		// Add promotion pieces
		const pieces = [QUEEN, ROOK, BISHOP, KNIGHT];
		for (const type of pieces) {
			const cacheKey = `${color}_${type}`;
			const pieceDiv = DOMUtils.createElement('div', {
				className: 'promotion-piece',
				attributes: { 'data-piece': type }
			});
			
			if (this.pieceCache?.[cacheKey]) {
				const originalImg = this.pieceCache[cacheKey];
				if (originalImg) {
					const clone = originalImg.cloneNode(true);
					clone.alt = type;
					DOMUtils.append(pieceDiv, clone);
				}
			} else {
				// Fallback to building URL if not cached
				const url = this._getPieceUrl(color, type);
				DOMUtils.append(pieceDiv, `<img src='${url}' alt='${type}'>`);
			}
			
			DOMUtils.append(panel, pieceDiv);
		}
		
		const cancelDiv = DOMUtils.createElement('div', { className: 'promotion-cancel', textContent: '✕' });
		DOMUtils.append(panel, cancelDiv);
		
		// Event handlers for both mouse and touch
		const handlePieceSelection = (e) => {
			const promotionPiece = e.target.closest('.promotion-piece');
			if (promotionPiece) {
				board._completePromotion(promotionPiece.dataset.piece);
				return;
			}

			const cancelEl = e.target.closest('.promotion-cancel');
			if (cancelEl) {
				board._cancelPromotion(fromSquareId);
			}
		};

		// Add both click and touch handlers
		DOMUtils.on(panel, 'click touchend', e => {
			if (e.type === 'touchend') {
				e.preventDefault();
			}
			handlePieceSelection(e);
		});
	}

	/**
	 * Completes the promotion with the selected piece type.
	 * @private
	 * @param {string} pieceType The type of piece to promote to (q, r, b, n).
	 * @return {Object|null} The move result or null if error.
	 */
	_completePromotion(pieceType, usermove = true) {
		if (!this.pendingPromotion) return;

		const {from, to, fromIdx, toIdx, animate, fenBefore} = this.pendingPromotion;
		const fromSquare = this.getSquare(fromIdx);
		const toSquare = this.getSquare(toIdx);
		const piece = DOMUtils.$('img', fromSquare);
		
		DOMUtils.remove(DOMUtils.$(`#promotion-panel-${this.id}`));
		this._updateBoard(fromSquare, toSquare, piece);
		if (fenBefore) this.chess.load(fenBefore);
		
		const move = this.chess.move({from, to, promotion: pieceType});
		if (!move) {
			console.error(`Illegal promotion move from ${from} to ${to}`);
			this.pendingPromotion = null;
			return null;
		}

		DOMUtils.empty(toSquare);
		this._createPiece(toIdx, pieceType, this.chess.turn() === WHITE ? BLACK : WHITE);
		
		if (animate) this._animatePiece(DOMUtils.$('img', toSquare), fromSquare);
		this._playSound(Sound.PROMOTE);

		// Emit promotion complete event
		this.emit('promotioncomplete', pieceType);
		this.onPromotionComplete?.(move);
		this.pendingPromotion = null;

		if (usermove) this.emit('usermove', move);

		return move;
	}

	/**
	 * Cancels the pending promotion.
	 * @private
	 * @param {number} fromSquareId The id of the square the pawn moved from.
	 */
	_cancelPromotion(fromSquareId) {
		// Remove promotion panel
		DOMUtils.remove(DOMUtils.$(`#promotion-panel-${this.id}`));
		
		// Notify parent component if callback exists
		if (typeof this.onPromotionComplete === 'function') {
			this.onPromotionComplete(null);
		}

		const currentSquare = this.getSquare(fromSquareId);
		const piece = DOMUtils.$('img', currentSquare);
		if(piece) DOMUtils.setStyle(piece, { top: '0px', left: '0px' });

		this.pendingPromotion = null;
	}

	/**
	 * Handles undoing promotion moves.
	 * @private
	 * @param {Object} moveResult - The move result object.
	 * @param {boolean} animate - Whether to animate the move.
	 * @return {boolean} True if promotion undo was handled; otherwise, false.
	 */
	_handleUndoPromotion(moveResult, animate = false) {
		if (moveResult.promotion) {
			// Get square indices for from and to
			const fromIndex = this.algebraicToIndex(moveResult.from, this.flipped);
			const toIndex = this.algebraicToIndex(moveResult.to, this.flipped);
			
			const fromSquare = this.getSquare(fromIndex);
			const toSquare = this.getSquare(toIndex);
			
			// Create a temporary pawn at the promotion square for animation
			if (animate) {
				// Create pawn of the appropriate color at the to square
				this._createPiece(toIndex, PAWN, moveResult.color);
				const pawn = DOMUtils.find(toSquare, 'img');
				
				// Now move this pawn back to the from square
				this._updateBoard(toSquare, fromSquare, pawn, false);
				this._animatePiece(pawn, toSquare);
			} else {
				// Without animation, just clear both squares
				DOMUtils.empty(fromSquare);
				DOMUtils.empty(toSquare);
				
				// Create pawn at the from square
				this._createPiece(fromIndex, PAWN, moveResult.color);
			}
			
			// If there was a captured piece, restore it at the to square
			if (moveResult.captured) {
				if (animate) {
					DOMUtils.empty(toSquare);
				}
				const capturedPieceColor = moveResult.color === WHITE ? BLACK : WHITE;
				this._createPiece(toIndex, moveResult.captured, capturedPieceColor);
				this._playSound(Sound.CAPTURE);
			}
			
			return true;
		}
		return false;
	}

	/**
	 * Loads the position (creates and positions the pieces) for the first time.
	 * @private
	 */
	_load() {
		for (const row of this.chess.board()) {
			for (const square of row) {
				if (square) {
					const index = this.algebraicToIndex(square.square, this.flipped);
					this._createPiece(index, square.type, square.color);
				}
			}
		}
		this._onResize();
	}

	/**
	 * Creates and initializes a draggable chess piece.
	 * @private
	 * @param {number} index The board index where the piece is placed.
	 * @param {string} type The type of the piece (e.g., 'pawn', 'knight').
	 * @param {string} color The color of the piece ('white' or 'black').
	 */
	_createPiece(index, type, color) {
		const square = this.getSquare(index);
		const cacheKey = `${color}_${type}`;
		
		if (this.pieceCache?.[cacheKey]) {
			const originalImg = this.pieceCache[cacheKey];
			if (originalImg) {
				DOMUtils.append(square, originalImg.cloneNode(true));
			} else {
				console.error(`Piece not found in cache for ${color}_${type}`);
			}
		} else {
			DOMUtils.append(square, `<img class='ui-widget-content' src='${this._getPieceUrl(color, type)}' alt='${type}'/>`);
		}
	}

	/**
	 * Updates the board by moving the piece and resetting square styles.
	 * @private
	 * @param {HTMLElement} from The source square.
	 * @param {HTMLElement} to The target square.
	 * @param {HTMLElement} piece The piece to move.
	 * @param {boolean} classes Whether to include highlighting classes.
	 */
	_updateBoard(from, to, piece, classes = true) {
		DOMUtils.empty(to);
		DOMUtils.append(to, piece);
		DOMUtils.setStyle(piece, { top: '0px', left: '0px' });
		this.clearBoardElements();
		this.clearBoardHighlights();
		if (classes) {
			DOMUtils.addClass(to, Css.JUST_MOVED);
			DOMUtils.addClass(from, Css.JUST_MOVED);
		}
	}

	/**
	 * Animates a piece's top and left to zero, so in practice it's animating the piece to the square it's on right now.
	 * When the point and click technique is used, the piece is teleported to the target square and then offset so it's back on the source square before animating.
	 * @private
	 * @param {HTMLElement} piece The piece to animate to top and left zero.
	 * @param {HTMLElement|object} from The source square or an object with top and left properties to start from.
	 * @param {number} duration The duration of the animation.
	 * @param {string} easing The easing function of the animation.
	 */
	_animatePiece(piece, from, duration = this.settings.pieceAnimationDuration, easing = this.settings.pieceAnimationEasing) {
		let offsetTop = 0;
		let offsetLeft = 0;

		if (from instanceof HTMLElement) {
			const fromRect = from.getBoundingClientRect();
			const pieceRect = piece.getBoundingClientRect();
			offsetTop = fromRect.top - pieceRect.top;
			offsetLeft = fromRect.left - pieceRect.left;
		} else {
			offsetTop = from.top;
			offsetLeft = from.left;
		}

		piece.style.transition = 'none';
		piece.style.transform = `translate(${offsetLeft}px, ${offsetTop}px)`;
		piece.offsetHeight; // Force reflow
		piece.style.transition = `transform ${duration}s ${easing}`;
		piece.style.transform = 'translate(0, 0)';
	}

	/**
	 * Renders arrows on the chessboard canvas.
	 * @private
	 */
	_render() {
		if (this.isDestroyed) return;
		
		if (!this.canvas) {
			const canvas = DOMUtils.$(`#overlay-${this.id}`);
			if (!canvas) return;

			this.canvas = canvas;
		}

		// Cache the canvas context instead of getting it every time
		if (!this.canvasContext) {
			this.canvasContext = this.canvas.getContext('2d');
		}
		
		const ctx = this.canvasContext;
		const squaresContainer = DOMUtils.$('#squares-' + this.id);
		const squareSize = squaresContainer.clientWidth / 8;
		const halfSquare = squareSize / 2;

		this.canvas.width = this.canvas.clientWidth;
		this.canvas.height = this.canvas.clientHeight;

		for (const arrow of this.arrows) {
			// Support both legacy array format [from, to] and new object format
			const from = Array.isArray(arrow) ? arrow[0] : arrow.from;
			const to = Array.isArray(arrow) ? arrow[1] : arrow.to;
			const color = Array.isArray(arrow) ? undefined : arrow.color;
			const opacity = Array.isArray(arrow) ? undefined : arrow.opacity;
			
			this._drawArrow(
				this.canvas, ctx,
				// Fancy (not really) math to get the squares center
				(from % 8) * squareSize + halfSquare, Math.floor(from / 8) * squareSize + halfSquare,
				(to % 8) * squareSize + halfSquare, Math.floor(to / 8) * squareSize + halfSquare,
				color, opacity
			);
		}
	}

	/**
	 * Handles certain static elements of the board when it's resized.
	 * @private
	 */
	_onResize() {
		if (this.isDestroyed) return;
		
		const board = DOMUtils.$("#squares-" + this.id);
		this.squareSize = DOMUtils.getOffset(board).width / 8;
	}

	/**
	 * Toggles the 'highlight' class on the given square element.
	 * @private
	 * @param {HTMLElement} square The jQuery object representing the square to be highlighted.
	 */
	_highlight(square) {
		DOMUtils.toggleClass(square, 'highlight');

		const index = this.getSquareIndex(square);
		const highlightIndex = this.highlights.indexOf(index);
		
		if (highlightIndex !== -1) {
			this.highlights.splice(highlightIndex, 1);
		} else {
			this.highlights.push(index);
		}
		
		const squareNotation = this.indexToAlgebraic(index, this.flipped);
		this.emit('highlight', squareNotation);
	}

	/**
	 * Draws an arrow on the given canvas.
	 * @private
	 * @param {HTMLCanvasElement} canvas The canvas element.
	 * @param {CanvasRenderingContext2D} ctx The rendering context.
	 * @param {number} fromX The starting x-coordinate.
	 * @param {number} fromY The starting y-coordinate.
	 * @param {number} toX The ending x-coordinate.
	 * @param {number} toY The ending y-coordinate.
	 * @param {string} color Optional custom color (with or without alpha).
	 * @param {number} opacity Optional opacity value (0-1).
	 */
	_drawArrow(canvas, ctx, fromX, fromY, toX, toY, color, opacity) {
		if (!canvas || !ctx) return;
		
		const s = canvas.width;
		const headLength = s / 16;
		ctx.lineWidth = s / 48;
		
		// Use custom color and opacity if provided
		let arrowColor = color || this.settings.styling.arrowColor;
		if (opacity !== undefined && opacity !== 1) {
			// If a color is provided without alpha, add the opacity
			// If color already has alpha (rgba), we'll need to parse and adjust it
			if (arrowColor.startsWith('rgba')) {
				// Replace existing alpha with new opacity
				arrowColor = arrowColor.replace(/[\d.]+\)$/, `${opacity})`);
			} else if (arrowColor.startsWith('rgb')) {
				// Convert rgb to rgba
				arrowColor = arrowColor.replace('rgb', 'rgba').replace(')', `, ${opacity})`);
			} else if (arrowColor.startsWith('#')) {
				// Convert hex to rgba
				const r = parseInt(arrowColor.slice(1, 3), 16);
				const g = parseInt(arrowColor.slice(3, 5), 16);
				const b = parseInt(arrowColor.slice(5, 7), 16);
				arrowColor = `rgba(${r}, ${g}, ${b}, ${opacity})`;
			}
		}
		ctx.fillStyle = ctx.strokeStyle = arrowColor;

		// If the move is knight move
		const threshold = 0.05;
		const knightRatio = Math.abs((fromX - toX) / (fromY - toY));
		const length = Math.sqrt((toX - fromX) ** 2 + (toY - fromY) ** 2);
		if ((length / s < 0.35) && 
			(Math.abs(knightRatio - 0.5) < threshold || 
				Math.abs(knightRatio - 2) < threshold)) {
			return this._drawKnightArrow(ctx, fromX, fromY, toX, toY, s, headLength);
		}

		const f = 0.865 * headLength;
		const angle = Math.atan2(toY - fromY, toX - fromX);
		const xOff = f * Math.cos(angle);
		const yOff = f * Math.sin(angle);

		const x1 = fromX + (s / 22) * Math.cos(angle);
		const y1 = fromY + (s / 22) * Math.sin(angle);
		const x2 = toX - xOff;
		const y2 = toY - yOff;
		
		ctx.beginPath();
		ctx.moveTo(x1, y1);
		ctx.lineTo(x2, y2);
		ctx.stroke();

		this._drawArrowhead(ctx, toX, toY, angle, headLength);
	}

	/**
	 * Draws an arrowhead on a canvas context pointing in a specified direction.
	 * The arrowhead is drawn at the target coordinates using two angled lines and filled.
	 * @private
	 * @param {CanvasRenderingContext2D} ctx - The 2D rendering context for the drawing surface.
	 * @param {number} toX - The x-coordinate of the arrow tip (the point the arrow is pointing to).
	 * @param {number} toY - The y-coordinate of the arrow tip.
	 * @param {number} angle - The angle (in radians) the arrow is pointing.
	 * @param {number} length - The length of each side of the arrowhead.
	 */
	_drawArrowhead(ctx, toX, toY, angle, length) {
		if (!ctx) return;

		ctx.beginPath();
		ctx.moveTo(toX, toY);
		for (const a of [-Math.PI / 6, Math.PI / 6]) {
			ctx.lineTo(
				toX - length * Math.cos(angle + a),
				toY - length * Math.sin(angle + a));
		}
		ctx.closePath();
		ctx.fill();
	}

	/**
	 * Draws an L-shaped arrow for knight moves on the given canvas.
	 * @private
	 * @param {CanvasRenderingContext2D} ctx The rendering context.
	 * @param {number} fromX The starting x-coordinate.
	 * @param {number} fromY The starting y-coordinate.
	 * @param {number} toX The ending x-coordinate.
	 * @param {number} toY The ending y-coordinate.
	 * @param {number} s The scale factor, which is just the canvas width.
	 * @param {number} headLength The length of the arrowhead.
	 */
	_drawKnightArrow(ctx, fromX, fromY, toX, toY, s, headLength) {
		if (!ctx) return;
		
		const dx = toX - fromX;
		const dy = toY - fromY;
		const horizontalFirst = Math.abs(dx) > Math.abs(dy);
		const cornerX = horizontalFirst ? toX : fromX;
		const cornerY = horizontalFirst ? fromY : toY;
		
		const dirX = Math.sign(dx) * (horizontalFirst ? -1 : 1);
		const dirY = Math.sign(dy) * (horizontalFirst ? -1 : 1);

		const f = 0.865 * headLength;
		const angle = Math.atan2(toY - cornerY, toX - cornerX);
		const xOff = f * Math.cos(angle);
		const yOff = f * Math.sin(angle);
		const offsetX = (s / 100) * dirX;
		const offsetY = (s / 100) * dirY;

		// Draw first segment of L
		const x1 = fromX - (horizontalFirst ? (s / 22) * dirX : 0);
		const y1 = fromY + (!horizontalFirst ? (s / 22) * dirY : 0);
		const x2 = !horizontalFirst ? cornerX : cornerX + offsetX;
		const y2 = horizontalFirst ? cornerY : cornerY + offsetY;
		const x3 = horizontalFirst ? cornerX : cornerX + offsetX;
		const y3 = !horizontalFirst ? cornerY : cornerY + offsetY;
		
		ctx.beginPath();
		ctx.moveTo(x1, y1);
		ctx.lineTo(x2, y2);
		ctx.stroke();
		
		ctx.beginPath();
		ctx.moveTo(x3, y3);
		ctx.lineTo(toX - xOff, toY - yOff);
		ctx.stroke();

		this._drawArrowhead(ctx, toX, toY, angle, headLength);
	}

	/**
	 * Adds a classification to the given square.
	 * @private
	 * @param {string|object} classification - The classification name or object.
	 * @param {HTMLElement} from - The jQuery object representing the starting square.
	 * @param {HTMLElement} to - The jQuery object representing the ending square
	 */
	addClassification(classification, from, to, showClassification = true) {
		let trueClassification;
		if (typeof classification === 'object') {
			trueClassification = classification;
		} else {
			trueClassification = Classification[classification.toUpperCase()];
		}
		if (!trueClassification) return;

		// If showClassification is false, only add the just-moved highlight without classification colors/icons
		if (!showClassification) {
			// Just add the just-moved class without classification-specific styling
			DOMUtils.addClass(to, Css.JUST_MOVED);
			DOMUtils.addClass(from, Css.JUST_MOVED);
			return;
		}

		DOMUtils.addClass(to, `board-classification ${trueClassification.class}`);
		DOMUtils.addClass(from, `board-classification ${trueClassification.class}`);

		const index = this.getSquareIndex(to);
		const col = index % 8;
		const onBorder = col === 7 || index < 8;

		if (trueClassification?.cachedImg) {
			const clone = trueClassification.cachedImg.cloneNode(true);
			DOMUtils.addClass(clone, 'classification');
			DOMUtils.removeClass(clone, 'move-icon'); // Keeping this in for parity reasons
			DOMUtils.addClass(clone, onBorder ? 'border' : '');
			DOMUtils.append(to, clone);
		} else {
			DOMUtils.append(to, `<img class='classification ${onBorder ? 'border' : ''}' src='${trueClassification.src}'>`);
		}
	}

	/**
	 * Initializes the input functionality for the game board.
	 * @private
	 */
	_initializeInput() {
		if (!this.settings.isInteractive || this.isDestroyed) return;

		const board = DOMUtils.$(`#squares-${this.id}`);
		if (!board) return;

		// Mouse and touch events use the same handlers
		this._addTrackedEventListener(board, 'mousedown', this.boundOnMouseDown);
		this._addTrackedEventListener(board, 'mousemove', this.boundOnMouseMove);
		this._addTrackedEventListener(board, 'mouseup', this.boundOnMouseUp);
		this._addTrackedEventListener(board, 'touchstart', this.boundOnMouseDown, { passive: false });
		this._addTrackedEventListener(board, 'touchmove', this.boundOnMouseMove, { passive: false });
		this._addTrackedEventListener(board, 'touchend', this.boundOnMouseUp);
	}

	/**
	 * @private
	 */
	_onMouseDown(event) {
		if (!this._canInteract()) return;

		const { x, y } = this._getEventCoordinates(event);
		const square = this.getSquareFromPosition(x, y);
		if (!square) return;

		this.dragStarted = false;
		this.dragStartXY = { x, y };

		// Does the square have a piece?
		if (!square.firstChild) {
			
			return;
		};

		event.preventDefault();

		if (event.type === 'touchstart' || event.button === 0) {
			this._startDrag(square, event);
		}
	}

	/**
	 * @private
	 */
	_onMouseMove(event, fromDocument = false) {
		if (!this._canInteract()) return;

		const { x, y } = this._getEventCoordinates(event);
		this._continueDrag(x, y);
	}

	/**
	 * @private
	 */
	_onMouseUp(event, fromDocument = false) {
		if (!this._canInteract()) return;

		const { x, y } = this._getEventCoordinates(event);

		if (event.type === 'touchend' || event.button === 0) {
			const wasClick = this._stopDrag(x, y);
			if (wasClick && !fromDocument) this._handleClickLogic(event);
		} else if (event.button === 2 && !fromDocument) {
			const startSquare = this.getSquareFromPosition(this.dragStartXY?.x, this.dragStartXY?.y);
			const endSquare = this.getSquareFromPosition(x, y);
			if (!startSquare || !endSquare) return;

			if (startSquare.dataset.square === endSquare.dataset.square) {
				this._highlight(startSquare);
			} else {
				this.createArrow(startSquare, endSquare);
			}
		}
	}

	/**
	 * @private
	 */
	_onDocumentMouseMove(event) {
		this._onMouseMove(event, true);
	}

	/**
	 * @private
	 */
	_onDocumentMouseUp(event) {
		const { x, y } = this._getEventCoordinates(event);
		this._stopDrag(x, y);
	}

	/**
	 * @private
	 */
	_startDrag(square, event) {
		if (!this.settings.isInteractive || this.isDestroyed) return;

		const index = this.getSquareIndex(square);
		const legalDestinations = this._getLegalDestinations(index);
		
		// Emit dragstart event - allow prevention
		const piece = square.firstChild;
		const squareNotation = this.indexToAlgebraic(index, this.flipped);
		const allowDrag = this.emit('dragstart', piece, squareNotation);
		
		// Check if drag was prevented by callback
		if (allowDrag === false) {
			return;
		}
		
		if (legalDestinations.length > 0) {
			DOMUtils.removeClass(DOMUtils.$$(`#squares-${this.id} .square`), Css.DROPPABLE);
			this.clearBoardElements();
			legalDestinations.forEach(index => DOMUtils.addClass(this.getSquare(index), Css.DROPPABLE));
		}

		this.dragPiece = square.firstChild;
		this.startDragIndex = index;

		const { x, y } = this._getEventCoordinates(event);
		this._continueDrag(x, y);

		// Add document listeners for dragging outside board
		['mousemove', 'touchmove'].forEach(type => this._addTrackedEventListener(document, type, this.boundDocumentMouseMove, { passive: false }));
		['mouseup', 'touchend'].forEach(type => this._addTrackedEventListener(document, type, this.boundDocumentMouseUp));
	}

	/**
	 * @private
	 */
	_continueDrag(x, y) {
		if (!this.settings.isInteractive || this.isDestroyed || !this.dragPiece) return;
		
		const start = this.dragStartXY;
		const dragDistance = Math.sqrt((x - start.x) ** 2 + (y - start.y) ** 2);

		if (dragDistance > this.settings.pieceDragThreshold) this.dragStarted = true;
		if (!this.dragStarted) return;

		// Constrain to board boundaries
		const boardRect = DOMUtils.$(`#squares-${this.id}`).getBoundingClientRect();
		const constrainedX = Math.max(boardRect.left, Math.min(x, boardRect.right));
		const constrainedY = Math.max(boardRect.top, Math.min(y, boardRect.bottom));

		this._movePieceToCursor(this.dragPiece, constrainedX, constrainedY);

		const square = this.getSquareFromPosition(constrainedX, constrainedY);
		if (this.hoveredSquare === square) return;

		DOMUtils.removeClass(this.hoveredSquare, Css.DROPPABLE_HOVER);
		DOMUtils.addClass(square, Css.DROPPABLE_HOVER);
		this.hoveredSquare = square;
		
		// Emit dragmove event
		if (square) {
			const squareNotation = this.indexToAlgebraic(this.getSquareIndex(square), this.flipped);
			this.emit('dragmove', this.dragPiece, squareNotation);
		}
	}

	/**
	 * @private
	 */
	_stopDrag(x, y) {
		if (!this.settings.isInteractive || this.isDestroyed || !this.dragPiece) return true;

		// Remove document listeners
		['mousemove', 'touchmove'].forEach(type => this._removeTrackedEventListener(document, type, this.boundDocumentMouseMove));
		['mouseup', 'touchend'].forEach(type => this._removeTrackedEventListener(document, type, this.boundDocumentMouseUp));

		const square = this.getSquareFromPosition(x, y);
		if (!square) {
			// Animated revert drops outside board bounds
			const fromSquare = this.getSquare(this.startDragIndex);
			this._animatePiece(this.dragPiece, {
				top: this.dragPiece.offsetTop - fromSquare.offsetTop,
				left: this.dragPiece.offsetLeft - fromSquare.offsetLeft
			}, this.settings.pieceRevertDuration, this.settings.pieceRevertEasing);
		
			this._deselectPiece();
		}

		const dragDistance = Math.sqrt((x - this.dragStartXY.x) ** 2 + (y - this.dragStartXY.y) ** 2);

		if (this.dragStarted && dragDistance > this.settings.pieceClickThreshold && square) {
			const toIndex = this.getSquareIndex(square);
			const fromSquare = this.indexToAlgebraic(this.startDragIndex, this.flipped);
			const toSquare = this.indexToAlgebraic(toIndex, this.flipped);
			
			// Emit drop event
			this.emit('drop', this.dragPiece, fromSquare, toSquare);
			
			if (this._attemptMove(this.startDragIndex, toIndex)) {
				this._deselectPiece();
			}
		}

		// Reset drag piece styling
		DOMUtils.setStyle(this.dragPiece, { height: 'unset', width: 'unset', top: '0px', left: '0px' });
		DOMUtils.removeClass(this.hoveredSquare, Css.DROPPABLE_HOVER);
		DOMUtils.removeClass(this.dragPiece, 'being-dragged');
		DOMUtils.removeClass(DOMUtils.$$(`#squares-${this.id} .square`), Css.DROPPABLE);

		this.dragPiece = undefined;
		this.hoveredSquare = undefined;

		return dragDistance <= this.settings.pieceClickThreshold;
	}

	/**
	 * @private
	 */
	_movePieceToCursor(piece, x, y) {
		const squareSize = this.squareSize;
		const offset = piece.offsetWidth / 2;

		DOMUtils.addClass(piece, 'being-dragged');
		DOMUtils.setStyle(piece, { 
			height: `${squareSize}px`, 
			width: `${squareSize}px`,
			top: `${y - offset}px`, 
			left: `${x - offset}px` 
		});
	}

	/**
	 * @private
	 */
	_deselectPiece() {
		DOMUtils.removeClass(DOMUtils.$$(`#squares-${this.id} .square`), `${Css.DROPPABLE} ${Css.SELECTED}`);
		this.selectedPiece = undefined;
	}

	/**
	 * Extracts coordinates from mouse or touch events
	 * @private
	 */
	_getEventCoordinates(event) {
		const { clientX = 0, clientY = 0 } = event.touches?.[0] || event.changedTouches?.[0] || event;
		return { x: clientX, y: clientY };
	}

	/**
	 * Shared click logic for both mouse and touch events
	 * @private
	 */
	_handleClickLogic(event) {
		if (!this._canInteract() || this.pendingPromotion !== null) return;

		const { x, y } = this._getEventCoordinates(event);
		const square = this.getSquareFromPosition(x, y);
		if (!square) return;
		
		const squareIndex = this.getSquareIndex(square);
		this.clearBoardElements();

		// Handle deselection
		if (this.selectedPiece === squareIndex) {
			return this._deselectPiece();
		}

		// Handle move attempt
		if (this.selectedPiece !== undefined && this._attemptMove(this.selectedPiece, squareIndex, true)) {
			return this._deselectPiece();
		}

		// Select new piece
		this._selectPieceIfValid(square, squareIndex);
	}

	/**
	 * Selects the piece on the board if it has any legal moves.
	 * @private
	 */
	_selectPieceIfValid(square, squareIndex) {
		if (this.isDestroyed || !DOMUtils.find(square, 'img')) {
			return this._deselectPiece();
		}

		DOMUtils.removeClass(DOMUtils.$$(`#squares-${this.id} .square`), `${Css.HIGHLIGHT} ${Css.DROPPABLE} ${Css.SELECTED}`);
		
		const legalDestinations = this._getLegalDestinations(squareIndex);
		if (legalDestinations.length === 0) {
			return this._deselectPiece();
		}
		
		DOMUtils.addClass(square, Css.SELECTED);
		legalDestinations.forEach(index => {
			const destSquare = this.getSquare(index);
			if (destSquare) DOMUtils.addClass(destSquare, Css.DROPPABLE);
		});
		
		this.selectedPiece = squareIndex;
		
		// Emit piece click event
		const piece = DOMUtils.find(square, 'img');
		const squareNotation = this.indexToAlgebraic(squareIndex, this.flipped);
		this.emit('piececlick', piece, squareNotation);
	}

	/**
	 * Loads and caches SVG piece images for faster rendering.
	 * @private
	 * @return {Promise} Promise that resolves when all pieces are cached
	 */
	async _cachePieces() {
		const loadPromises = [];
		for (const color of [WHITE, BLACK]) {
			for (const type of [KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN]) {
				const key = `${color}_${type}`;
				const url = this._getPieceUrl(color, type);
				
				loadPromises.push(
					fetch(url)
						.then(response => response.ok ? response.text() : Promise.reject(response.status))
						.then(svgText => {
							const dataUri = `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svgText)))}`;
							this.pieceCache[key] = DOMUtils.createElement('img', {
								attributes: { src: dataUri, alt: type }
							});
						})
						.catch(error => console.error(`Failed to load ${url}:`, error))
				);
			}
		}
		await Promise.allSettled(loadPromises);
	}

	/**
	 * Loads and caches audio files for move sounds.
	 * First checks for custom URLs, then falls back to folder-based naming schemes.
	 * @private
	 * @return {Promise} Promise that resolves when all audio is cached
	 */
	async _cacheAudio() {
		this.volumeNode.gain.value = 0.5;
		this.volumeNode.connect(this.audioContext.destination);
		
		for (const sound of Object.values(Sound)) {
			const soundUrl = this.settings.theme.customSoundUrls?.[sound] || 
				`${this.settings.theme.soundFoldersPath}/${this.settings.theme.soundFolderName}/${sound}.mp3`;
			
			try {
				const resp = await fetch(soundUrl);
				if (!resp.ok) throw new Error(`HTTP error! status: ${resp.status}`);
				const array = await resp.arrayBuffer();
				this.audioBuffers[sound] = await this.audioContext.decodeAudioData(array);
			} catch (error) {
				console.error(`Failed to load sound ${sound} from ${soundUrl}:`, error);
			}
		}
	}

	/**
	 * Builds the URL for a piece image based on the current theme settings.
	 * First checks for custom URLs, then falls back to folder-based naming schemes.
	 * @private
	 * @param {string} color - The piece color ('w' or 'b')
	 * @param {string} type - The piece type ('p', 'n', 'b', 'r', 'q', 'k')
	 * @returns {string} The complete URL to the piece image
	 */
	_getPieceUrl(color, type) {
		const pieceKey = `${color}_${type}`;
		if (this.settings.theme.customPieceUrls?.[pieceKey]) {
			return this.settings.theme.customPieceUrls[pieceKey];
		}
		
		const { pieceFoldersPath, pieceFolderName, pieceFormat } = this.settings.theme;
		return pieceFormat === 'standard' 
			? `${pieceFoldersPath}/${pieceFolderName}/${color}/${type}.svg`
			: `${pieceFoldersPath}/${pieceFolderName}/${color}${type.toUpperCase()}.svg`;
	}

	/**
	 * Play a sound effect
	 * @private
	 * @param {string} sound The name of the sound that will be played.
	 */
	_playSound(sound) {
		if (this.isDestroyed || !this.settings.audioEnabled) return;
		
		if (this.audioContext.state === 'suspended') this.audioContext.resume();
		if (!this.audioBuffers[sound]) return;
		const src = this.audioContext.createBufferSource();
		
		src.buffer = this.audioBuffers[sound];
		src.connect(this.volumeNode);
		src.start(0);
	}

	/**
	 * Plays a sound effect based on the outcome of the move.
	 * @private
	 * @param {Object} moveResult The move result object.
	 */
	_playSoundBasedOnOutcome(moveResult) {
		if (this.isDestroyed) return;
		
		if (this.chess.inCheck()) {
			this._playSound(Sound.CHECK);
		} else if (moveResult.isCapture()) {
			this._playSound(Sound.CAPTURE);
		} else {
			this._playSound(Sound.MOVE);
		}
	}

	/**
	 * Returns the DOM object for a square element by its index.
	 * @private
	 * @param {number} squareIndex The index of the square.
	 * @return {HTMLElement} The DOM object for the square.
	 */
	getSquare(squareIndex) {
		return this.squares?.[squareIndex] || 
			DOMUtils.$(`#squares-${this.id} .square[data-square='${squareIndex}']`);
	}

	/**
	 * Returns the board index (0-63) from a given DOM square element.
	 * @private
	 * @param {HTMLElement} square The DOM object for the square.
	 * @return {number} The index of the square.
	 */
	getSquareIndex(square) {
		return square ? parseInt(square.dataset.square, 10) : -1;
	}

	/**
	 * Returns the DOM object for a square element by its position.
	 * @private
	 * @param {number} x The x-coordinate of the position.
	 * @param {number} y The y-coordinate of the position.
	 * @return {HTMLElement|null} The DOM object for the square.
	 */
	getSquareFromPosition(x, y) {
		if (!x || !y) return null;
		const squaresContainer = DOMUtils.$(`#squares-${this.id}`);
		if (!squaresContainer) return null;

		for (const square of DOMUtils.$$('.square', squaresContainer)) {
			const bounds = square.getBoundingClientRect();
			if (x >= bounds.left && x <= bounds.left + bounds.width &&
				y >= bounds.top && y <= bounds.top + bounds.height) {
				return square;
			}
		}
		return null;
	}

	/**
	 * Converts a board index (0-63) to algebraic chess notation (e.g., 'e4').
	 * @private
	 * @param {number} index The board index (0-63), where 0 is a1 and 63 is h8.
	 * @param {boolean} flip Whether to flip the board for black's perspective.
	 * @return {string} The algebraic notation of the square (e.g., 'e4').
	 * @throws {Error} If the index is out of range.
	 */
	indexToAlgebraic(index, flip = false) {
		if (index < 0 || index > 63) throw new Error('Invalid index. Index must be between 0 and 63.');
		if (flip) index = 63 - index;
		return String.fromCharCode(97 + (index % 8)) + (8 - Math.floor(index / 8));
	}

	/**
	 * Converts algebraic chess notation (e.g., 'e4') to a board index (0-63).
	 * @private
	 * @param {string} notation The algebraic notation (e.g., 'e4').
	 * @param {boolean} flip Whether to flip the board useful for black's perspective.
	 * @return {number} The board index corresponding to the notation.
	 * @throws {Error} If the notation is not in the correct format.
	 */
	algebraicToIndex(notation, flip = false) {
		if (!/^[a-h][1-8]$/.test(notation)) {
			throw new Error('Invalid chess notation. Expected format: [a-h][1-8]');
		}
		const column = notation.charCodeAt(0) - 97;
		const row = 8 - parseInt(notation.charAt(1), 10);
		const final = row * 8 + column;
		return flip ? 63 - final : final;
	}

	/**
	 * Creates and injects the dynamic CSS stylesheet for this chessboard instance
	 * @private
	 */
	_createStyleSheet() {
		this.styleSheet = DOMUtils.createStyleSheet(`chessboard-styles-${this.id}`);
		this._injectBaseStyles();
		this._injectAnimations();
	}

	/**
	 * Injects all base CSS styles for the chessboard
	 * @private
	 */
	_injectBaseStyles() {
		const s = this.settings.styling;
		const theme = this.settings.theme;

		// Chessboard container
		DOMUtils.addCSSRule(this.styleSheet, `#${this.id}`, {
			'display': 'flex',
			'flex-direction': 'row',
			'align-items': 'center',
			'position': 'relative',
			'font-family': s.fontFamily
		});

		// Board squares container
		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id}`, {
			'width': '100%',
			'height': '100%',
			'aspect-ratio': '1 / 1',
			'max-width': '100%',
			'max-height': '100%',
			'display': 'grid',
			'grid-template-columns': 'repeat(8, minmax(0, 5fr))',
			'grid-template-rows': 'repeat(8, minmax(0, 5fr))',
			'padding': '0px',
			'margin': '0px',
			'overflow': 'hidden',
			'background-image': theme.boardImageBackground ? `url("${theme.boardBackgroundPath}")` : 'none',
			'background-size': 'cover',
			'background-repeat': 'no-repeat',
			'background-position': 'center center',
			'user-select': 'none',
			'-webkit-touch-callout': 'none',
			'-webkit-text-size-adjust': 'none',
			'-webkit-user-select': 'none',
			'font-family': s.fontFamily
		});

		// Board overlay (for arrows)
		DOMUtils.addCSSRule(this.styleSheet, `#overlay-${this.id}`, {
			'position': 'absolute',
			'display': 'block',
			'width': '100%',
			'height': '100%',
			'aspect-ratio': '1 / 1',
			'max-width': '100%',
			'max-height': '100%',
			'pointer-events': 'none',
			'z-index': '10'
		});

		// Square base styles
		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .square`, {
			'width': '100%',
			'height': '100%',
			'display': 'flex',
			'position': 'relative'
		});

		// Square colors
		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .square.dark`, {
			'background-color': theme.boardDarkSquareColor
		});

		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .square.light`, {
			'background-color': theme.boardLightSquareColor
		});

		// Squares that have pieces
		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .square:has(> img)`, {
			'cursor': s.hoverCursor
		});

		// Cursor for when any piece is being dragged
		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id}:has(img.being-dragged) .square`, {
			'cursor': s.grabCursor
		});

		// Piece styling
		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .square img`, {
			'pointer-events': 'none',
			'z-index': '2'
		});

		// Dragged piece styling
		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .square .being-dragged`, {
			'position': 'fixed',
			'scale': s.draggedPieceScale.toString(),
			'z-index': '100'
		});

		// Board states
		this._injectBoardStates();

		// Promotion panel
		this._injectPromotionStyles();

		// Classification styles
		this._injectClassificationStyles();

		// Board labels (notation)
		this._injectBoardLabelStyles();
	}

	/**
	 * Injects CSS for different board states (selected, highlighted, etc.)
	 * @private
	 */
	_injectBoardStates() {
		const s = this.settings.styling;

		// Selected square
		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .selected-square, #squares-${this.id} .square:has(> img.being-dragged)`, {
			'background-image': `linear-gradient(${s.selectedSquareColor} 100%, ${s.selectedSquareColor} 0%)`,
			'cursor': s.grabCursor
		});

		// Just moved squares
		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id}.just-moved .square`, {
			'background-image': `linear-gradient(${s.justMovedColor} 100%, ${s.justMovedColor} 0%)`
		});

		// Droppable squares
		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .ui-droppable-active:not(.board-classification)`, {
			'background-image': `radial-gradient(${s.droppableIndicatorColor} 23%, transparent 23%)`
		});

		// Combined droppable and square has piece (enemy piece)
		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .ui-droppable-active:has(> img)::before`, {
			'content': '""',
			'position': 'absolute',
			'top': '0',
			'left': '0',
			'right': '0',
			'bottom': '0',
			'border': `${s.captureIndicatorSize} solid ${s.captureIndicatorColor}`,
			'border-radius': '50%',
			'z-index': '2',
			'pointer-events': 'none'
		});

		// Droppable hover
		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .ui-droppable-hover`, {
			'box-shadow': `0 0 0 min(${s.droppableHoverBorderWidth}, ${s.droppableHoverBorderWidth}) ${s.droppableHoverBorderColor} inset`
		});

		// Highlighted squares
		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .highlight`, {
			'background-image': `linear-gradient(${s.highlightColor} 100%, ${s.highlightColor} 0%)`
		});

		// Draggable styling
		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .ui-draggable`, {
			'user-select': 'none',
			'z-index': '9'
		});
	}

	/**
	 * Injects CSS for promotion panel
	 * @private
	 */
	_injectPromotionStyles() {
		const s = this.settings.styling;

		DOMUtils.addCSSRule(this.styleSheet, `#promotion-panel-${this.id}`, {
			'position': 'absolute',
			'background-color': s.promotionPanelBackground,
			'box-shadow': s.promotionPanelShadow,
			'display': 'flex',
			'flex-direction': 'column',
			'z-index': '15',
			'color': 'white',
			'backdrop-filter': 'blur(3px)',
			'overflow': 'hidden',
			'animation': 'fadeIn 0.2s ease-out',
			'width': '100%',
			'user-select': 'none',
			'-webkit-touch-callout': 'none',
			'-webkit-text-size-adjust': 'none',
			'-webkit-user-select': 'none'
		});

		DOMUtils.addCSSRule(this.styleSheet, `#promotion-panel-${this.id}.bottom`, {
			'flex-direction': 'column-reverse'
		});

		DOMUtils.addCSSRule(this.styleSheet, `#promotion-panel-${this.id} .promotion-piece`, {
			'display': 'flex',
			'align-items': 'center',
			'justify-content': 'center',
			'cursor': 'pointer',
			'transition': 'background-color 0.2s'
		});

		DOMUtils.addCSSRule(this.styleSheet, `#promotion-panel-${this.id} .promotion-piece:hover`, {
			'background-color': s.promotionPieceHoverColor
		});

		DOMUtils.addCSSRule(this.styleSheet, `#promotion-panel-${this.id} .promotion-piece img`, {
			'width': '100%',
			'height': '100%',
			'pointer-events': 'none'
		});

		DOMUtils.addCSSRule(this.styleSheet, `#promotion-panel-${this.id} .promotion-cancel`, {
			'height': s.promotionCancelHeight,
			'display': 'flex',
			'align-items': 'center',
			'justify-content': 'center',
			'cursor': 'pointer',
			'background-color': s.promotionCancelBackground,
			'transition': 'background-color 0.2s',
			'font-size': '18px',
			'font-weight': 'bold'
		});

		DOMUtils.addCSSRule(this.styleSheet, `#promotion-panel-${this.id} .promotion-cancel:hover`, {
			'background-color': s.promotionCancelHoverBackground
		});
	}

	/**
	 * Injects CSS for move classifications
	 * @private
	 */
	_injectClassificationStyles() {
		const s = this.settings.styling;

		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .classification`, {
			'position': 'absolute',
			'width': s.classificationSize,
			'height': s.classificationSize,
			'transform': `translate(${s.classificationOffsetX}, ${s.classificationOffsetY})`,
			'z-index': '11',
			'pointer-events': 'none'
		});

		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .classification.border`, {
			'transform': `translate(${s.classificationBorderOffsetX}, ${s.classificationBorderOffsetY})`
		});
	}

	/**
	 * Injects CSS animations
	 * @private
	 */
	_injectAnimations() {
		DOMUtils.injectKeyframes(this.styleSheet, 'fadeIn', {
			'from': { 'opacity': '0' },
			'to': { 'opacity': '1' }
		});
	}

	/**
	 * Injects CSS for board labels (rank and file notation)
	 * @private
	 */
	_injectBoardLabelStyles() {
		if (!this.settings.showBoardLabels) return;

		const s = this.settings.styling;
		const theme = this.settings.theme;

		// Common styles for all notation labels
		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .square::after`, {
			'position': 'absolute',
			'font-size': s.notationFontSize,
			'font-weight': s.notationFontWeight,
			'color': theme.boardLightSquareColor
		});

		// Left edge - rank numbers (every 8th square starting from 0)
		const rankSquares = Array.from({length: 8}, (_, i) => `#squares-${this.id} .square[data-square="${i * 8}"]::after`).join(', ');
		DOMUtils.addCSSRule(this.styleSheet, rankSquares, {
			'left': s.notationOffset,
			'top': s.notationOffset
		});

		// Bottom edge - file letters (squares 56-63)
		const fileSquares = Array.from({length: 8}, (_, i) => `#squares-${this.id} .square[data-square="${56 + i}"]::after`).join(', ');
		DOMUtils.addCSSRule(this.styleSheet, fileSquares, {
			'right': '4px',
			'bottom': s.notationOffset
		});

		// a1 needs both rank and file
		const aFileLabel = this.flipped ? 'h' : 'a';
		DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .square[data-square="56"]::before`, {
			'content': `"${aFileLabel}"`,
			'position': 'absolute',
			'right': s.notationOffset,
			'bottom': s.notationOffset,
			'font-size': s.notationFontSize,
			'font-weight': s.notationFontWeight,
			'color': theme.boardLightSquareColor
		});

		// Individual rank and file labels
		const ranks = this.flipped ? ['1', '2', '3', '4', '5', '6', '7', '8'] : ['8', '7', '6', '5', '4', '3', '2', '1'];
		const files = this.flipped ? ['g', 'f', 'e', 'd', 'c', 'b', 'a'] : ['b', 'c', 'd', 'e', 'f', 'g', 'h']; // first file handled separately above

		ranks.forEach((rank, i) => {
			DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .square[data-square="${i * 8}"]::after`, {
				'content': `"${rank}"`,
				'color': i % 2 === 0 ? theme.boardDarkSquareColor : theme.boardLightSquareColor
			});
		});

		files.forEach((file, i) => {
			DOMUtils.addCSSRule(this.styleSheet, `#squares-${this.id} .square[data-square="${57 + i}"]::after`, {
				'content': `"${file}"`,
				'color': i % 2 === 0 ? theme.boardDarkSquareColor : theme.boardLightSquareColor
			});
		});
	}

	/**
	 * Updates board labels after flipping
	 * @private
	 */
	_updateBoardLabels() {
		if (!this.settings.showBoardLabels) return;
		
		// Recreate the entire stylesheet to update labels
		this._createStyleSheet();
	}
}