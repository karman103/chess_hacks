export class SettingsMenu {
    constructor(containerSelector, chessUI = null) {
        this.container = document.querySelector(containerSelector);
        this.chessUI = chessUI;
        this.eventListeners = [];
        this.settingsConfig = this._getDefaultSettingsConfig();
        this.cookiePrefix = 'centichess_setting_';
        this.cookieExpireDays = 365;
        
        // Performance optimization: Cache frequently used DOM elements
        this.cachedElements = new Map();
        
        // Throttle input events for better performance
        this.throttledInputHandler = this._throttle(this._handleInputEvent.bind(this), 16); // ~60fps
    }

    init(chessboard) {
        this.chessboard = chessboard;
        this.render();
        this.bindEvents();
        this.loadSettingsFromCookies();
    }

    /**
     * Default settings configuration - purely data-driven
     */
    _getDefaultSettingsConfig() {
        return {
            'Board Appearance': {
                settings: {
                    'boardTheme': {
                        type: 'visual-list',
                        label: 'Board Theme',
                        description: 'Choose your preferred board style',
                        defaultValue: 'green',
                        path: 'theme.boardColors',
                        options: [
                            { 
                                value: 'classic', 
                                label: 'Classic',
                                preview: { type: 'board', light: '#f0d9b5', dark: '#b58863' },
                                actions: [
                                    { path: 'theme.boardLightSquareColor', value: '#f0d9b5' },
                                    { path: 'theme.boardDarkSquareColor', value: '#b58863' }
                                ]
                            },
                            { 
                                value: 'green', 
                                label: 'Green',
                                preview: { type: 'board', light: '#e0e0e0', dark: '#6ea176' },
                                actions: [
                                    { path: 'theme.boardLightSquareColor', value: '#e0e0e0' },
                                    { path: 'theme.boardDarkSquareColor', value: '#6ea176' }
                                ]
                            },
                            { 
                                value: 'blue', 
                                label: 'Blue',
                                preview: { type: 'board', light: '#dee3e6', dark: '#8ca2ad' },
                                actions: [
                                    { path: 'theme.boardLightSquareColor', value: '#dee3e6' },
                                    { path: 'theme.boardDarkSquareColor', value: '#8ca2ad' }
                                ]
                            },
                            { 
                                value: 'purple', 
                                label: 'Purple',
                                preview: { type: 'board', light: '#e8e4f0', dark: '#9b7aa0' },
                                actions: [
                                    { path: 'theme.boardLightSquareColor', value: '#e8e4f0' },
                                    { path: 'theme.boardDarkSquareColor', value: '#9b7aa0' }
                                ]
                            },
                            { 
                                value: 'red', 
                                label: 'Red',
                                preview: { type: 'board', light: '#ffe6e6', dark: '#cc5555' },
                                actions: [
                                    { path: 'theme.boardLightSquareColor', value: '#ffe6e6' },
                                    { path: 'theme.boardDarkSquareColor', value: '#cc5555' }
                                ]
                            },
                            { 
                                value: 'orange', 
                                label: 'Orange',
                                preview: { type: 'board', light: '#ffe6e6', dark: '#cd8042' },
                                actions: [
                                    { path: 'theme.boardLightSquareColor', value: '#ffe6e6' },
                                    { path: 'theme.boardDarkSquareColor', value: '#cd8042' }
                                ]
                            }
                        ]
                    },
                    'pieceTheme': {
                        type: 'visual-list',
                        label: 'Piece Set',
                        description: 'Choose your preferred piece style',
                        defaultValue: 'cburnett',
                        path: 'theme.pieceFolderName',
                        options: [
                            { 
                                value: 'cburnett', 
                                label: 'Classic', 
                                preview: { type: 'pieces', folder: 'cburnett' }
                            },
                            { 
                                value: 'alpha', 
                                label: 'Alpha', 
                                preview: { type: 'pieces', folder: 'alpha' }
                            },
                            { 
                                value: 'anarcandy', 
                                label: 'Anarcandy', 
                                preview: { type: 'pieces', folder: 'anarcandy' }
                            },
                            { 
                                value: 'caliente', 
                                label: 'Caliente', 
                                preview: { type: 'pieces', folder: 'caliente' }
                            },
                            { 
                                value: 'california', 
                                label: 'California', 
                                preview: { type: 'pieces', folder: 'california' }
                            },
                            { 
                                value: 'cardinal', 
                                label: 'Cardinal', 
                                preview: { type: 'pieces', folder: 'cardinal' }
                            },
                            { 
                                value: 'celtic', 
                                label: 'Celtic', 
                                preview: { type: 'pieces', folder: 'celtic' }
                            },
                            { 
                                value: 'chess7', 
                                label: 'Chess7', 
                                preview: { type: 'pieces', folder: 'chess7' }
                            },
                            { 
                                value: 'chessnut', 
                                label: 'Chessnut', 
                                preview: { type: 'pieces', folder: 'chessnut' }
                            },
                            { 
                                value: 'companion', 
                                label: 'Companion', 
                                preview: { type: 'pieces', folder: 'companion' }
                            },
                            { 
                                value: 'cooke', 
                                label: 'Cooke', 
                                preview: { type: 'pieces', folder: 'cooke' }
                            },
                            { 
                                value: 'dubrovny', 
                                label: 'Dubrovny', 
                                preview: { type: 'pieces', folder: 'dubrovny' }
                            },
                            { 
                                value: 'fantasy', 
                                label: 'Fantasy', 
                                preview: { type: 'pieces', folder: 'fantasy' }
                            },
                            { 
                                value: 'firi', 
                                label: 'Firi', 
                                preview: { type: 'pieces', folder: 'firi' }
                            },
                            { 
                                value: 'fresca', 
                                label: 'Fresca', 
                                preview: { type: 'pieces', folder: 'fresca' }
                            },
                            { 
                                value: 'gioco', 
                                label: 'Gioco', 
                                preview: { type: 'pieces', folder: 'gioco' }
                            },
                            { 
                                value: 'governor', 
                                label: 'Governor', 
                                preview: { type: 'pieces', folder: 'governor' }
                            },
                            { 
                                value: 'horsey', 
                                label: 'Horsey', 
                                preview: { type: 'pieces', folder: 'horsey' }
                            },
                            { 
                                value: 'icpieces', 
                                label: 'IC Pieces', 
                                preview: { type: 'pieces', folder: 'icpieces' }
                            },
                            { 
                                value: 'kiwen-suwi', 
                                label: 'Kiwen Suwi', 
                                preview: { type: 'pieces', folder: 'kiwen-suwi' }
                            },
                            { 
                                value: 'kosal', 
                                label: 'Kosal', 
                                preview: { type: 'pieces', folder: 'kosal' }
                            },
                            { 
                                value: 'leipzig', 
                                label: 'Leipzig', 
                                preview: { type: 'pieces', folder: 'leipzig' }
                            },
                            { 
                                value: 'maestro', 
                                label: 'Maestro', 
                                preview: { type: 'pieces', folder: 'maestro' }
                            },
                            { 
                                value: 'merida', 
                                label: 'Merida', 
                                preview: { type: 'pieces', folder: 'merida' }
                            },
                            { 
                                value: 'monarchy', 
                                label: 'Monarchy', 
                                preview: { type: 'pieces', folder: 'monarchy' }
                            },
                            { 
                                value: 'mpchess', 
                                label: 'MP Chess', 
                                preview: { type: 'pieces', folder: 'mpchess' }
                            },
                            { 
                                value: 'pirouetti', 
                                label: 'Pirouetti', 
                                preview: { type: 'pieces', folder: 'pirouetti' }
                            },
                            { 
                                value: 'pixel', 
                                label: 'Pixel', 
                                preview: { type: 'pieces', folder: 'pixel' }
                            },
                            { 
                                value: 'reillycraig', 
                                label: 'Reilly Craig', 
                                preview: { type: 'pieces', folder: 'reillycraig' }
                            },
                            { 
                                value: 'rhosgfx', 
                                label: 'Rhos GFX', 
                                preview: { type: 'pieces', folder: 'rhosgfx' }
                            },
                            { 
                                value: 'riohacha', 
                                label: 'Riohacha', 
                                preview: { type: 'pieces', folder: 'riohacha' }
                            },
                            { 
                                value: 'shapes', 
                                label: 'Shapes', 
                                preview: { type: 'pieces', folder: 'shapes' }
                            },
                            { 
                                value: 'spatial', 
                                label: 'Spatial', 
                                preview: { type: 'pieces', folder: 'spatial' }
                            },
                            { 
                                value: 'staunty', 
                                label: 'Staunty', 
                                preview: { type: 'pieces', folder: 'staunty' }
                            },
                            { 
                                value: 'tatiana', 
                                label: 'Tatiana', 
                                preview: { type: 'pieces', folder: 'tatiana' }
                            },
                            { 
                                value: 'chesshacks', 
                                label: 'Chess Hacks', 
                                preview: { type: 'pieces', folder: 'chesshacks' }
                            }
                        ]
                    },
                    'boardColors': {
                        type: 'group',
                        label: 'Custom Board Colors',
                        description: 'Customize light and dark square colors manually',
                        settings: [
                            {
                                key: 'boardLightSquareColor',
                                type: 'color',
                                label: 'Light Squares',
                                defaultValue: '#f0d9b5',
                                path: 'theme.boardLightSquareColor',
                                preview: { target: '.color-preview-box' }
                            },
                            {
                                key: 'boardDarkSquareColor',
                                type: 'color',
                                label: 'Dark Squares',
                                defaultValue: '#b58863',
                                path: 'theme.boardDarkSquareColor',
                                preview: { target: '.color-preview-box' }
                            }
                        ]
                    },
                    // 'boardSettings': {
                    //     type: 'group',
                    //     label: 'Board Settings',
                    //     description: 'Board settings',
                    //     settings: [
                    //         //pieceAnimationDuration
                    //         {
                    //             key: 'pieceAnimationDuration',
                    //             type: 'slider',
                    //             label: 'Piece Animation Duration',
                    //             defaultValue: 0.1,
                    //             min: 0,
                    //             max: 0.5,
                    //             step: 0.02,
                    //             path: 'pieceAnimationDuration'
                    //         }
                    //     ]
                    // }
                }
            },
            'Engine & Quick Settings': {
                settings: {
                    'engineSettings': {
                        type: 'group',
                        label: 'Engine Settings',
                        description: 'Engine settings',
                        settings: [
                            {
                                key: 'showClassification',
                                type: 'toggle',
                                label: 'Show Classifications',
                                description: 'Show classifications for each move',
                                defaultValue: true,
                                path: 'showClassification'
                            },
                            {
                                key: 'showBestMoveArrow',
                                type: 'toggle',
                                label: 'Show Best Move Arrow',
                                description: 'Show a arrow pointing to the best move',
                                defaultValue: false,
                                path: 'showBestMoveArrow'
                            },
                            {
                                key: 'showProbabilityArrows',
                                type: 'toggle',
                                label: 'Show Probability Arrows',
                                description: 'Show arrows for move probabilities on the board',
                                defaultValue: true,
                                path: 'showProbabilityArrows'
                            },
                            {
                                key: 'engineType',
                                type: 'dropdown',
                                label: 'Engine Type',
                                defaultValue: 'stockfish-17-lite',
                                options: [
                                    {
                                        value: 'stockfish-17-lite',
                                        label: 'Stockfish 17 Lite'
                                    },
                                    {
                                        value: 'stockfish-16-nnue',
                                        label: 'Stockfish 16 NNUE '
                                    },
                                    {
                                        value: 'stockfish-16-lite',
                                        label: 'Stockfish 16 Lite'
                                    },
                                    {
                                        value: 'stockfish-11',
                                        label: 'Stockfish 11'
                                    }
                                ],
                            },
                            {
                                key: 'engineDepth',
                                type: 'slider',
                                label: 'Engine Depth',
                                defaultValue: 14,
                                min: 2,
                                max: 24,
                            },
                            {
                                key: 'variationEngineDepth',
                                type: 'slider',
                                label: 'Variation Engine Depth',
                                defaultValue: 14,
                                min: 2,
                                max: 24,
                            }
                        ]
                    },
                    'quickToggles': {
                        type: 'group',
                        label: 'Quick Toggles',
                        description: 'Quickly toggle settings',
                        settings: [
                            {
                                key: 'showBoardLabels',
                                type: 'toggle',
                                label: 'Show Board Labels',
                                description: 'Display file and rank labels around the board',
                                defaultValue: true,
                                path: 'showBoardLabels'
                            },
                            {
                                key: 'audioEnabled',
                                type: 'toggle',
                                label: 'Sound Effects',
                                description: 'Play sounds for moves and captures',
                                defaultValue: true,
                                path: 'audioEnabled'
                            }
                        ]
                    }
                }
            }
        };
    }

    /**
     * Add a new settings group
     */
    addSettingsGroup(groupName, groupConfig) {
        this.settingsConfig[groupName] = groupConfig;
        this.render();
    }

    /**
     * Add a setting to an existing group
     */
    addSetting(groupName, settingKey, settingConfig) {
        if (this.settingsConfig[groupName]) {
            this.settingsConfig[groupName].settings[settingKey] = settingConfig;
            this.render();
        }
    }

    /**
     * Render the entire settings menu
     */
    render() {
        this.container.innerHTML = '';
        this._clearElementCache();
        
        Object.entries(this.settingsConfig).forEach(([groupName, groupConfig]) => {
            const groupElement = this._createSettingsGroup(groupName, groupConfig);
            this.container.appendChild(groupElement);
        });
    }

    /**
     * Create a settings group container
     */
    _createSettingsGroup(groupName, groupConfig) {
        const groupDiv = document.createElement('div');
        groupDiv.className = 'settings-group';
        
        const header = document.createElement('div');
        header.className = 'settings-group-header';
        header.innerHTML = `<h3 class="settings-group-title">${groupName}</h3>`;
        
        const content = document.createElement('div');
        content.className = 'settings-group-content';
        
        Object.entries(groupConfig.settings).forEach(([settingKey, settingConfig]) => {
            const settingElement = this._createSetting(settingKey, settingConfig);
            content.appendChild(settingElement);
        });
        
        groupDiv.appendChild(header);
        groupDiv.appendChild(content);
        
        return groupDiv;
    }

    /**
     * Create individual setting based on type
     */
    _createSetting(settingKey, config) {
        if (config.type === 'group') {
            return this._createGroupedSetting(settingKey, config);
        }

        const settingDiv = document.createElement('div');
        settingDiv.className = 'setting-item';
        settingDiv.setAttribute('data-setting', settingKey);
        
        const label = document.createElement('div');
        label.className = 'setting-label';
        label.innerHTML = `
            <span class="setting-name">${config.label}</span>
            ${config.description ? `<span class="setting-description">${config.description}</span>` : ''}
        `;
        
        const input = document.createElement('div');
        input.className = 'setting-input';
        
        const inputElement = this._createInputByType(settingKey, config);
        input.appendChild(inputElement);
        
        settingDiv.appendChild(label);
        settingDiv.appendChild(input);
        
        return settingDiv;
    }

    /**
     * Create input element based on type - generic factory method
     */
    _createInputByType(settingKey, config) {
        const inputCreators = {
            'color': () => this._createColorInput(settingKey, config),
            'slider': () => this._createSliderInput(settingKey, config),
            'dropdown': () => this._createDropdownInput(settingKey, config),
            'visual-list': () => this._createVisualListInput(settingKey, config),
            'toggle': () => this._createToggleInput(settingKey, config),
            'button': () => this._createButtonInput(settingKey, config)
        };

        const creator = inputCreators[config.type];
        if (creator) {
            return creator();
        } else {
            const errorSpan = document.createElement('span');
            errorSpan.className = 'setting-error';
            errorSpan.textContent = `Unknown setting type: ${config.type}`;
            return errorSpan;
        }
    }

    /**
     * Create grouped setting container
     */
    _createGroupedSetting(groupKey, config) {
        const groupDiv = document.createElement('div');
        groupDiv.className = 'setting-item setting-group-item';
        groupDiv.setAttribute('data-setting', groupKey);
        
        if (config.label) {
            const label = document.createElement('div');
            label.className = 'setting-group-label';
            label.innerHTML = `
                <span class="setting-name">${config.label}</span>
                ${config.description ? `<span class="setting-description">${config.description}</span>` : ''}
            `;
            groupDiv.appendChild(label);
        }
        
        const groupContent = document.createElement('div');
        groupContent.className = 'setting-group-content-inline';
        
        config.settings.forEach(subConfig => {
            const subSettingDiv = document.createElement('div');
            subSettingDiv.className = 'setting-sub-item';
            subSettingDiv.setAttribute('data-setting', subConfig.key);
            
            const subLabel = document.createElement('div');
            subLabel.className = 'setting-sub-label';
            subLabel.innerHTML = `<span class="setting-sub-name">${subConfig.label}</span>`;
            
            const subInput = document.createElement('div');
            subInput.className = 'setting-sub-input';
            
            const inputElement = this._createInputByType(subConfig.key, subConfig);
            subInput.appendChild(inputElement);
            
            subSettingDiv.appendChild(subLabel);
            subSettingDiv.appendChild(subInput);
            groupContent.appendChild(subSettingDiv);
        });
        
        groupDiv.appendChild(groupContent);
        return groupDiv;
    }

    /**
     * Create color picker input
     */
    _createColorInput(settingKey, config) {
        const container = document.createElement('div');
        container.className = 'color-input-container';
        
        const currentValue = this.getSettingValue(settingKey) || config.defaultValue;
        
        const colorBox = document.createElement('div');
        colorBox.className = 'color-preview-box';
        colorBox.style.backgroundColor = currentValue;
        
        const input = document.createElement('input');
        input.type = 'color';
        input.className = 'color-input';
        input.value = currentValue;
        input.setAttribute('data-path', config.path);
        input.setAttribute('data-setting-key', settingKey);
        
        container.appendChild(colorBox);
        container.appendChild(input);
        
        return container;
    }

    /**
     * Create slider input
     */
    _createSliderInput(settingKey, config) {
        const container = document.createElement('div');
        container.className = 'slider-input-container';
        
        const currentValue = this.getSettingValue(settingKey) !== null ? this.getSettingValue(settingKey) : config.defaultValue;
        
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.className = 'slider-input';
        slider.min = config.min;
        slider.max = config.max;
        slider.step = config.step;
        slider.value = currentValue;
        slider.setAttribute('data-path', config.path);
        slider.setAttribute('data-setting-key', settingKey);
        
        // Set initial progress for blue left-side track
        this._updateSliderProgress(slider, currentValue, config.min, config.max);
        
        const valueDisplay = document.createElement('span');
        valueDisplay.className = 'slider-value';
        valueDisplay.textContent = config.format ? config.format(currentValue) : currentValue;
        
        container.appendChild(slider);
        container.appendChild(valueDisplay);
        
        return container;
    }

    /**
     * Create dropdown input
     */
    _createDropdownInput(settingKey, config) {
        const select = document.createElement('select');
        select.className = 'dropdown-input';
        select.setAttribute('data-path', config.path);
        select.setAttribute('data-setting-key', settingKey);
        
        const currentValue = this.getSettingValue(settingKey) || config.defaultValue;
        
        config.options.forEach(option => {
            const optionElement = document.createElement('option');
            optionElement.value = option.value;
            optionElement.textContent = option.label;
            if (option.value === currentValue) {
                optionElement.selected = true;
            }
            select.appendChild(optionElement);
        });
        
        return select;
    }

    /**
     * Create visual list input - completely data-driven preview generation
     */
    _createVisualListInput(settingKey, config) {
        const container = document.createElement('div');
        container.className = 'visual-list-container';
        
        const currentValue = this.getSettingValue(settingKey) || config.defaultValue;
        
        config.options.forEach(option => {
            const optionDiv = document.createElement('div');
            optionDiv.className = 'visual-list-option';
            optionDiv.setAttribute('data-value', option.value);
            optionDiv.setAttribute('data-path', config.path);
            optionDiv.setAttribute('data-setting-key', settingKey);
            
            if (option.value === currentValue) {
                optionDiv.classList.add('selected');
            }
            
            // Generate preview based on preview config
            const previewElement = this._createPreviewElement(option.preview);
            const labelElement = document.createElement('span');
            labelElement.className = 'visual-list-label';
            labelElement.textContent = option.label;
            
            optionDiv.appendChild(previewElement);
            optionDiv.appendChild(labelElement);
            container.appendChild(optionDiv);
        });
        
        return container;
    }

    /**
     * Create preview element based on preview configuration
     */
    _createPreviewElement(previewConfig) {
        if (!previewConfig) {
            return document.createElement('div');
        }

        const previewCreators = {
            'board': (config) => {
                const preview = document.createElement('div');
                preview.className = 'board-preview';
                preview.innerHTML = `
                    <div class="board-preview-square light" style="background-color: ${config.light}"></div>
                    <div class="board-preview-square dark" style="background-color: ${config.dark}"></div>
                    <div class="board-preview-square dark" style="background-color: ${config.dark}"></div>
                    <div class="board-preview-square light" style="background-color: ${config.light}"></div>
                `;
                return preview;
            },
            'pieces': (config) => {
                const preview = document.createElement('div');
                preview.className = 'piece-preview';
                preview.innerHTML = `
                    <img src="/pieces/${config.folder}/wK.svg" alt="White King" class="piece-preview-icon">
                    <img src="/pieces/${config.folder}/bQ.svg" alt="Black Queen" class="piece-preview-icon">
                `;
                return preview;
            }
        };

        const creator = previewCreators[previewConfig.type];
        return creator ? creator(previewConfig) : document.createElement('div');
    }

    /**
     * Create toggle input
     */
    _createToggleInput(settingKey, config) {
        const container = document.createElement('div');
        container.className = 'toggle-input-container';
        
        const currentValue = this.getSettingValue(settingKey) !== null ? this.getSettingValue(settingKey) : config.defaultValue;
        
        const toggle = document.createElement('input');
        toggle.type = 'checkbox';
        toggle.className = 'toggle-input';
        toggle.checked = currentValue;
        toggle.setAttribute('data-path', config.path);
        toggle.setAttribute('data-setting-key', settingKey);
        toggle.id = `toggle-${settingKey}`;
        
        const label = document.createElement('label');
        label.className = 'toggle-label';
        label.setAttribute('for', `toggle-${settingKey}`);
        
        const slider = document.createElement('span');
        slider.className = 'toggle-slider';
        
        label.appendChild(slider);
        container.appendChild(toggle);
        container.appendChild(label);
        
        return container;
    }

    /**
     * Create button input
     */
    _createButtonInput(settingKey, config) {
        const button = document.createElement('button');
        button.className = 'setting-button';
        button.textContent = config.buttonText || config.label;
        button.type = 'button';
        button.setAttribute('data-action', config.action || '');
        
        return button;
    }

    /**
     * Bind event listeners - optimized approach using event delegation
     */
    bindEvents() {
        this._cleanupEventListeners();

        // Use event delegation instead of binding to individual elements
        const containerHandler = (e) => this._handleContainerEvent(e);
        this.container.addEventListener('click', containerHandler);
        this.container.addEventListener('change', containerHandler);
        this.container.addEventListener('input', containerHandler);
        
        this.eventListeners.push({ 
            element: this.container, 
            event: 'click', 
            handler: containerHandler 
        });
        this.eventListeners.push({ 
            element: this.container, 
            event: 'change', 
            handler: containerHandler 
        });
        this.eventListeners.push({ 
            element: this.container, 
            event: 'input', 
            handler: containerHandler 
        });
    }

    /**
     * Centralized event handler using event delegation
     */
    _handleContainerEvent(e) {
        const element = e.target;
        const eventType = e.type;
        
        // Check if element is a settings control
        if (!this._isSettingsControl(element)) {
            return;
        }

        // Handle different event types
        if (eventType === 'input') {
            // Throttle input events for performance
            this.throttledInputHandler(e);
        } else if (eventType === 'change') {
            this._handleChangeEvent(e);
        } else if (eventType === 'click') {
            this._handleClickEvent(e);
        }
    }

    /**
     * Check if element is a settings control
     */
    _isSettingsControl(element) {
        return element.classList.contains('color-input') ||
               element.classList.contains('slider-input') ||
               element.classList.contains('dropdown-input') ||
               element.classList.contains('toggle-input') ||
               element.classList.contains('visual-list-option') ||
               element.classList.contains('setting-button') ||
               element.closest('.visual-list-option');
    }

    /**
     * Handle input events (throttled)
     */
    _handleInputEvent(e) {
        const element = e.target;
        const value = this._extractValueFromElement(element);
        
        // Handle preview updates for input events
        if (this._hasPreviewMode(element)) {
            this._handlePreviewUpdate(element, value);
        }
    }

    /**
     * Handle change events
     */
    _handleChangeEvent(e) {
        const element = e.target;
        const settingKey = element.getAttribute('data-setting-key');
        const path = element.getAttribute('data-path');
        const value = this._extractValueFromElement(element);
        
        if (settingKey && path) {
            this._handleValueChange(settingKey, path, value, element);
        } else {
            console.warn('Missing settingKey or path for element:', element, {
                settingKey: settingKey,
                path: path
            });
        }
    }

    /**
     * Handle click events
     */
    _handleClickEvent(e) {
        const element = e.target;
        const action = element.getAttribute('data-action');
        
        // Handle button actions
        if (element.classList.contains('setting-button') && action) {
            this._executeAction(action);
            return;
        }

        // Handle visual list clicks
        const visualListOption = element.classList.contains('visual-list-option') ? 
                                element : element.closest('.visual-list-option');
        if (visualListOption) {
            this._handleVisualListSelection(visualListOption);
        }
    }

    /**
     * Handle preview updates during input events - optimized and fixed
     */
    _handlePreviewUpdate(element, value) {
        if (element.classList.contains('color-input')) {
            // Use more specific targeting for color preview updates
            const container = element.closest('.color-input-container');
            const settingKey = element.getAttribute('data-setting-key');
            
            // Use setting-specific cache key to avoid conflicts
            let preview = this._getCachedElement('.color-preview-box', container);
            if (!preview) {
                preview = container.querySelector('.color-preview-box');
                if (preview && settingKey) {
                    this._cacheElement(`preview_${settingKey}`, preview);
                }
            }
            if (preview) {
                preview.style.backgroundColor = value;
            }
        } else if (element.classList.contains('slider-input')) {
            const container = element.closest('.slider-input-container');
            const settingKey = element.getAttribute('data-setting-key');
            
            // Use setting-specific cache key to avoid conflicts
            let valueDisplay = this._getCachedElement('.slider-value', container);
            if (!valueDisplay) {
                valueDisplay = container.querySelector('.slider-value');
                if (valueDisplay && settingKey) {
                    this._cacheElement(`slider_${settingKey}`, valueDisplay);
                }
            }
            
            if (valueDisplay) {
                const config = this._findSettingConfig(settingKey);
                valueDisplay.textContent = config?.format ? config.format(value) : value;
                
                // Update slider progress for blue left-side track
                if (config) {
                    this._updateSliderProgress(element, value, config.min, config.max);
                }
            }
        }
    }

    /**
     * Update the UI element for a specific setting - optimized and fixed
     */
    _updateSettingUI(settingKey, value) {
        // Use more specific targeting for settings
        let element = this._getCachedElement(`[data-setting-key="${settingKey}"]`);
        if (!element) {
            element = this.container.querySelector(`[data-setting-key="${settingKey}"]`);
            if (element) {
                this._cacheElement(`setting_${settingKey}`, element);
            }
        }
        
        if (!element) return;

        if (element.type === 'color') {
            element.value = value;
            const container = element.closest('.color-input-container');
            
            // Use setting-specific cache key and direct container targeting
            let preview = this.cachedElements.get(`preview_${settingKey}`);
            if (!preview) {
                preview = container?.querySelector('.color-preview-box');
                if (preview) {
                    this._cacheElement(`preview_${settingKey}`, preview);
                }
            }
            if (preview) {
                preview.style.backgroundColor = value;
            }
        } else if (element.type === 'checkbox') {
            element.checked = value;
        } else if (element.type === 'range') {
            element.value = value;
            const container = element.closest('.slider-input-container');
            
            // Use setting-specific cache key
            let valueDisplay = this.cachedElements.get(`slider_${settingKey}`);
            if (!valueDisplay) {
                valueDisplay = container?.querySelector('.slider-value');
                if (valueDisplay) {
                    this._cacheElement(`slider_${settingKey}`, valueDisplay);
                }
            }
            const config = this._findSettingConfig(settingKey);
            if (valueDisplay) {
                valueDisplay.textContent = config?.format ? config.format(value) : value;
            }
            // Update slider progress for blue left-side track
            if (config) {
                this._updateSliderProgress(element, value, config.min, config.max);
            }
        } else if (element.tagName === 'SELECT') {
            element.value = value;
        }
    }

    /**
     * Handle visual list selection - optimized
     */
    _handleVisualListSelection(option) {
        const container = option.closest('.visual-list-container');
        const path = option.getAttribute('data-path');
        const value = option.getAttribute('data-value');
        const settingKey = option.getAttribute('data-setting-key');
        
        // Batch DOM updates to prevent layout thrashing
        requestAnimationFrame(() => {
            // Update selection
            const options = container.querySelectorAll('.visual-list-option');
            options.forEach(opt => opt.classList.remove('selected'));
            option.classList.add('selected');
        });
        
        this._handleValueChange(settingKey, path, value, option);
    }

    /**
     * Handle deselection of visual-list options - optimized
     */
    _handleVisualListDeselection(changedPath, changedValue) {
        // Use more efficient DOM queries
        const visualListContainers = this.container.querySelectorAll('.visual-list-container');
        
        // Find visual-list settings that have actions affecting this path
        this._traverseSettings((settingKey, config) => {
            if (config.type === 'visual-list' && config.options) {
                // Check if any option's actions would set this path
                const hasMatchingAction = config.options.some(option => 
                    option.actions && option.actions.some(action => action.path === changedPath)
                );
                
                if (hasMatchingAction) {
                    // Check if current selection's actions still match all related paths
                    const currentValue = this.getSettingValue(settingKey);
                    const currentOption = this._findOptionByValue(config, currentValue);
                    
                    if (currentOption && currentOption.actions) {
                        const stillMatches = currentOption.actions.every(action => {
                            if (action.path === changedPath) {
                                return action.value === changedValue;
                            }
                            // Check if other related paths still match
                            const relatedValue = this.getSettingValue(this._findSettingKeyByPath(action.path));
                            return relatedValue === action.value;
                        });
                        
                        if (!stillMatches) {
                            // Use cached element if available
                            let visualContainer = this._getCachedElement(`[data-setting-key="${settingKey}"] .visual-list-container`);
                            if (!visualContainer) {
                                const container = this.container.querySelector(`[data-setting-key="${settingKey}"]`);
                                visualContainer = container?.querySelector('.visual-list-container');
                            }
                            
                            if (visualContainer) {
                                // Batch DOM updates
                                requestAnimationFrame(() => {
                                    visualContainer.querySelectorAll('.visual-list-option').forEach(opt => 
                                        opt.classList.remove('selected')
                                    );
                                });
                                // Clear the saved value for the visual list setting
                                this._deleteCookie(settingKey);
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Clean up event listeners and cached elements
     */
    _cleanupEventListeners() {
        this.eventListeners.forEach(({ element, event, handler }) => {
            element.removeEventListener(event, handler);
        });
        this.eventListeners = [];
        this._clearElementCache();
    }

    /**
     * Cleanup
     */
    destroy() {
        this._cleanupEventListeners();
        this._clearElementCache();
    }

    /**
     * Throttle function to limit how often a function can be called
     */
    _throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        }
    }

    /**
     * Cache DOM element for reuse
     */
    _cacheElement(key, element) {
        this.cachedElements.set(key, element);
        return element;
    }

    /**
     * Get cached DOM element or query and cache it
     */
    _getCachedElement(selector, parent = this.container) {
        // Create a more unique cache key using parent element's data attributes or position
        let parentKey = 'container';
        if (parent !== this.container) {
            // Try to get a unique identifier from the parent
            const settingKey = parent.closest('[data-setting]')?.getAttribute('data-setting') ||
                              parent.querySelector('[data-setting-key]')?.getAttribute('data-setting-key') ||
                              'unknown';
            parentKey = `parent_${settingKey}`;
        }
        
        const key = `${parentKey}_${selector}`;
        if (this.cachedElements.has(key)) {
            return this.cachedElements.get(key);
        }
        const element = parent.querySelector(selector);
        if (element) {
            this._cacheElement(key, element);
        }
        return element;
    }

    /**
     * Clear element cache when re-rendering
     */
    _clearElementCache() {
        this.cachedElements.clear();
    }

    /**
     * Execute predefined actions
     */
    _executeAction(actionName) {
        const actions = {
            'clearAllSettings': () => this.clearAllSettings()
        };
        
        const action = actions[actionName];
        if (action) {
            action();
        }
    }

    /**
     * Update chessboard settings
     */
    _updateChessboardSetting(path, value) {
        if (this.chessboard && this.chessboard.setOption) {
            this.chessboard.setOption(path, value);
        }
    }

    /**
     * Find setting config by key
     */
    _findSettingConfig(settingKey) {
        for (const group of Object.values(this.settingsConfig)) {
            if (group.settings[settingKey]) {
                return group.settings[settingKey];
            }
            // Check grouped settings
            for (const setting of Object.values(group.settings)) {
                if (setting.type === 'group' && setting.settings) {
                    const found = setting.settings.find(s => s.key === settingKey);
                    if (found) return found;
                }
            }
        }
        return null;
    }

    /**
     * Update slider progress for blue left-side track
     */
    _updateSliderProgress(sliderElement, value, min, max) {
        const percentage = ((value - min) / (max - min)) * 100;
        sliderElement.style.setProperty('--slider-progress', `${percentage}%`);
    }

    /**
     * Find setting key by path
     */
    _findSettingKeyByPath(path) {
        let foundKey = null;
        this._traverseSettings((settingKey, config) => {
            if (config.path === path) {
                foundKey = settingKey;
            }
        });
        return foundKey;
    }

    /**
     * Find option by value in config
     */
    _findOptionByValue(config, value) {
        return config?.options?.find(option => option.value === value);
    }

    /**
     * Load all settings from cookies and apply them
     */
    loadSettingsFromCookies() {
        this._traverseSettings((settingKey, config) => {
            const savedValue = this._getCookie(settingKey);
            if (savedValue !== null) {
                const option = this._findOptionByValue(config, savedValue);
                
                if (option && option.actions) {
                    // Execute multiple actions
                    option.actions.forEach(action => {
                        this._updateChessboardSetting(action.path, action.value);
                    });
                } else if (config.path) {
                    // Single path update
                    this._updateChessboardSetting(config.path, savedValue);
                }
                
                // Update default value for UI
                config.defaultValue = savedValue;
            }
        });
    }

    /**
     * Save a setting to cookies
     */
    saveSettingToCookie(settingKey, value) {
        this._setCookie(settingKey, value);
    }

    /**
     * Force reapply all saved settings
     */
    reapplyAllSettings() {
        this.loadSettingsFromCookies();
    }

    /**
     * Clear all settings cookies
     */
    clearAllSettings() {
        this._traverseSettings((settingKey) => {
            this._deleteCookie(settingKey);
        });
        
        if (confirm('This will reset all settings to defaults and reload the page. Continue?')) {
            window.location.reload();
        }
    }

    /**
     * Traverse all settings in the config
     */
    _traverseSettings(callback) {
        Object.values(this.settingsConfig).forEach(groupConfig => {
            Object.entries(groupConfig.settings).forEach(([settingKey, settingConfig]) => {
                if (settingConfig.type === 'group') {
                    settingConfig.settings.forEach(subConfig => {
                        callback(subConfig.key, subConfig);
                    });
                } else {
                    callback(settingKey, settingConfig);
                }
            });
        });
    }

    /**
     * Get current setting value
     */
    getSettingValue(settingKey) {
        const savedValue = this._getCookie(settingKey);
        if (savedValue !== null) {
            return savedValue;
        }

        const config = this._findSettingConfig(settingKey);
        return config?.defaultValue || null;
    }

    /**
     * Cookie management methods
     */
    _setCookie(name, value, days = this.cookieExpireDays) {
        const expires = new Date();
        expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));
        document.cookie = `${this.cookiePrefix}${name}=${encodeURIComponent(JSON.stringify(value))};expires=${expires.toUTCString()};path=/`;
    }

    _getCookie(name) {
        const cookieName = `${this.cookiePrefix}${name}=`;
        const decodedCookie = decodeURIComponent(document.cookie);
        const cookies = decodedCookie.split(';');
        
        for (let cookie of cookies) {
            cookie = cookie.trim();
            if (cookie.indexOf(cookieName) === 0) {
                try {
                    return JSON.parse(cookie.substring(cookieName.length));
                } catch (e) {
                    console.warn(`Failed to parse cookie ${name}:`, e);
                    return null;
                }
            }
        }
        return null;
    }

    _deleteCookie(name) {
        document.cookie = `${this.cookiePrefix}${name}=;expires=Thu, 01 Jan 1970 00:00:00 UTC;path=/`;
    }

    /**
     * Extract value from different input types
     */
    _extractValueFromElement(element) {
        if (element.type === 'checkbox') {
            return element.checked;
        } else if (element.type === 'range') {
            return parseFloat(element.value);
        } else {
            return element.value;
        }
    }

    /**
     * Check if element has preview mode (should handle input events separately)
     */
    _hasPreviewMode(element) {
        return element.classList.contains('color-input') || element.classList.contains('slider-input');
    }

    /**
     * Handle final value changes - fixed to ensure proper updates
     */
    _handleValueChange(settingKey, path, value, element) {
        if (settingKey) {
            this.saveSettingToCookie(settingKey, value);
        }

        // Check for actions (multi-path updates for visual lists)
        const config = this._findSettingConfig(settingKey);
        const option = this._findOptionByValue(config, value);
        
        if (option && option.actions) {
            // Execute multiple actions (for board theme presets)
            option.actions.forEach(action => {
                this._updateChessboardSetting(action.path, action.value);
                
                // Find the setting key for this path and save to cookie
                const relatedSettingKey = this._findSettingKeyByPath(action.path);
                if (relatedSettingKey) {
                    this.saveSettingToCookie(relatedSettingKey, action.value);
                }
            });
            // Sync related settings after actions are applied
            this._syncRelatedSettings(option.actions.map(action => ({ path: action.path, value: action.value })));
        } else if (path) {
            // Single path update (for individual color/setting changes)
            this._updateChessboardSetting(path, value);
            // Sync related settings for single path update
            this._syncRelatedSettings([{ path, value }]);
        }

        // Handle showClassification toggle
        if (settingKey === 'showClassification' && this.chessUI) {
            this.chessUI.refreshClassificationDisplay();
        }

        // Handle showBestMoveArrow toggle
        if (settingKey === 'showBestMoveArrow' && this.chessUI) {
            this.chessUI.refreshBestMoveArrow();
        }

        // Handle showProbabilityArrows toggle
        if (settingKey === 'showProbabilityArrows' && this.chessUI) {
            this.chessUI.refreshProbabilityArrows();
        }
    }

    /**
     * Sync related settings that share the same paths - improved debugging
     */
    _syncRelatedSettings(changedPaths) {
        changedPaths.forEach(({ path, value }) => {
            // Find all settings that use this same path
            this._traverseSettings((settingKey, config) => {
                if (config.path === path) {
                    // Update the UI element for this setting
                    this._updateSettingUI(settingKey, value);
                    // Update the cookie to keep it in sync
                    this.saveSettingToCookie(settingKey, value);
                }
            });
            
            // Handle visual-list deselection when individual settings change
            this._handleVisualListDeselection(path, value);
        });
    }
}