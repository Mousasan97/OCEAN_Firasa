// OCEAN Personality Analysis App
// Use relative URL for deployment, or localhost for development
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8001/api/v1'
    : '/api/v1';

// Recording duration for quick record mode (in seconds)
const RECORDING_DURATION = 20;

// Stage mapping for progress updates
const STAGE_MAP = {
    'uploading': { message: 'Uploading video...', stageId: 'stageUpload' },
    'compressing': { message: 'Compressing video...', stageId: 'stageUpload' },
    'extracting_frames': { message: 'Extracting video frames...', stageId: 'stageFrames' },
    'transcribing': { message: 'Transcribing audio...', stageId: 'stageTranscribe' },
    'analyzing_video': { message: 'Analyzing facial expressions...', stageId: 'stageAnalyze' },
    'analyzing_audio': { message: 'Analyzing voice patterns...', stageId: 'stageAnalyze' },
    'generating_report': { message: 'Generating personality insights...', stageId: 'stageReport' },
    'complete': { message: 'Analysis complete!', stageId: null }
};

// Stage order for completion tracking
const STAGE_ORDER = ['stageUpload', 'stageFrames', 'stageTranscribe', 'stageAnalyze', 'stageReport'];

/**
 * Initialize mouse gradient effect for interactive background
 */
function initMouseGradient() {
    const gradientOrb = document.createElement('div');
    gradientOrb.className = 'mouse-gradient-orb';
    document.body.appendChild(gradientOrb);

    let mouseX = 0;
    let mouseY = 0;
    let currentX = 0;
    let currentY = 0;

    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });

    function animateGradient() {
        // Smooth follow with easing
        currentX += (mouseX - currentX) * 0.1;
        currentY += (mouseY - currentY) * 0.1;

        gradientOrb.style.left = currentX + 'px';
        gradientOrb.style.top = currentY + 'px';

        requestAnimationFrame(animateGradient);
    }

    animateGradient();
}

/**
 * Toggle mobile navigation menu
 */
function toggleMobileMenu() {
    const mobileMenu = document.getElementById('mobileMenu');
    if (mobileMenu) {
        mobileMenu.classList.toggle('active');
    }
}

/**
 * Animate title word by word
 */
function animateTitle(element) {
    if (!element || element.classList.contains('animated')) return;

    const text = element.textContent.trim();
    const words = text.split(' ');

    element.innerHTML = '';
    element.classList.add('animated-title', 'animated');

    words.forEach((word, index) => {
        const span = document.createElement('span');
        span.className = 'animated-word';
        span.textContent = word;
        span.style.animationDelay = `${0.3 + index * 0.1}s`;
        element.appendChild(span);
    });
}

/**
 * Initialize all animations on DOM load
 */
function initAnimations() {
    // Animate upload page title
    const uploadTitle = document.querySelector('.upload-page-title');
    if (uploadTitle) {
        animateTitle(uploadTitle);
    }
}

// Initialize effects on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initMouseGradient();
    initAnimations();
});

/**
 * Initialize results navigation tabs with Vercel-style animated indicators
 */
function initResultsNavigation() {
    const nav = document.getElementById('resultsNav');
    const navItems = document.querySelectorAll('.results-nav-item');
    const hoverHighlight = document.getElementById('navHoverHighlight');
    const activeIndicator = document.getElementById('navActiveIndicator');

    if (!nav || !hoverHighlight || !activeIndicator) return;

    // Update hover highlight position
    function updateHoverHighlight(item) {
        if (!item) {
            hoverHighlight.style.opacity = '0';
            return;
        }
        const rect = item.getBoundingClientRect();
        const navRect = nav.getBoundingClientRect();

        hoverHighlight.style.left = `${rect.left - navRect.left}px`;
        hoverHighlight.style.top = `${rect.top - navRect.top}px`;
        hoverHighlight.style.width = `${rect.width}px`;
        hoverHighlight.style.height = `${rect.height}px`;
        hoverHighlight.style.opacity = '1';
    }

    // Update active indicator position
    function updateActiveIndicator(item) {
        if (!item) return;
        const rect = item.getBoundingClientRect();
        const navRect = nav.getBoundingClientRect();

        activeIndicator.style.left = `${rect.left - navRect.left}px`;
        activeIndicator.style.width = `${rect.width}px`;
    }

    // Initialize active indicator position
    const initialActive = document.querySelector('.results-nav-item.active');
    if (initialActive) {
        // Small delay to ensure layout is calculated
        setTimeout(() => updateActiveIndicator(initialActive), 100);
    }

    // Hover events for highlight
    navItems.forEach(item => {
        item.addEventListener('mouseenter', () => {
            updateHoverHighlight(item);
        });
    });

    nav.addEventListener('mouseleave', () => {
        hoverHighlight.style.opacity = '0';
    });

    // Click handlers for navigation tabs
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const sectionId = item.getAttribute('data-section');
            const section = document.getElementById(sectionId);

            if (section) {
                // Update active state
                navItems.forEach(nav => nav.classList.remove('active'));
                item.classList.add('active');

                // Update active indicator with animation
                updateActiveIndicator(item);

                // Scroll to section
                section.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });

    // Scroll spy to highlight active nav item
    const observerOptions = {
        root: null,
        rootMargin: '-100px 0px -50% 0px',
        threshold: 0
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const sectionId = entry.target.id;
                navItems.forEach(navItem => {
                    const isActive = navItem.getAttribute('data-section') === sectionId;
                    navItem.classList.toggle('active', isActive);
                    if (isActive) {
                        updateActiveIndicator(navItem);
                    }
                });
            }
        });
    }, observerOptions);

    // Observe all sections
    document.querySelectorAll('.results-content-section, .section-card-new').forEach(section => {
        if (section.id) {
            observer.observe(section);
        }
    });

    // Update indicator on window resize
    window.addEventListener('resize', () => {
        const activeItem = document.querySelector('.results-nav-item.active');
        if (activeItem) {
            updateActiveIndicator(activeItem);
        }
    });
}

// Shareable cards carousel state
let currentShareableCardIndex = 0;

/**
 * Switch shareable tab
 */
function switchShareableTab(tabId) {
    const tabs = document.querySelectorAll('.shareable-tab');
    tabs.forEach(tab => {
        tab.classList.toggle('active', tab.getAttribute('data-tab') === tabId);
    });
}

/**
 * Go to previous shareable card
 */
function prevShareableCard() {
    const cards = document.querySelectorAll('.share-card');
    if (cards.length === 0) return;

    currentShareableCardIndex = (currentShareableCardIndex - 1 + cards.length) % cards.length;
    updateShareableCarousel();
}

/**
 * Go to next shareable card
 */
function nextShareableCard() {
    const cards = document.querySelectorAll('.share-card');
    if (cards.length === 0) return;

    currentShareableCardIndex = (currentShareableCardIndex + 1) % cards.length;
    updateShareableCarousel();
}

/**
 * Go to specific shareable card
 */
function goToShareableCard(index) {
    currentShareableCardIndex = index;
    updateShareableCarousel();
}

/**
 * Update carousel display - now shows all cards at once
 */
function updateShareableCarousel() {
    const cards = document.querySelectorAll('.share-card');

    // Show all cards (no longer a carousel, displays all 3 cards in a row)
    cards.forEach((card) => {
        card.style.display = 'flex';
    });

    // Hide carousel navigation since we show all cards
    const carouselArrows = document.querySelectorAll('.carousel-arrow');
    const carouselDots = document.querySelector('.carousel-dots');

    carouselArrows.forEach(arrow => arrow.style.display = 'none');
    if (carouselDots) carouselDots.style.display = 'none';
}

/**
 * Update progress UI with stage and percentage
 */
function updateProgress(stage, progress, message) {
    const progressFill = document.getElementById('progressFill');
    const progressPercent = document.getElementById('progressPercent');
    const progressStage = document.getElementById('progressStage');
    const loadingMessage = document.getElementById('loadingMessage');

    // Update progress bar
    if (progressFill) progressFill.style.width = `${progress}%`;
    if (progressPercent) progressPercent.textContent = `${progress}%`;

    // Update message
    const stageInfo = STAGE_MAP[stage];
    if (progressStage) progressStage.textContent = stageInfo?.message || message;
    if (loadingMessage) loadingMessage.textContent = message;

    // Update stage indicators
    if (stageInfo?.stageId) {
        const currentStageIndex = STAGE_ORDER.indexOf(stageInfo.stageId);

        STAGE_ORDER.forEach((id, index) => {
            const elem = document.getElementById(id);
            if (!elem) return;

            elem.classList.remove('active', 'completed');
            if (index < currentStageIndex) {
                elem.classList.add('completed');
            } else if (index === currentStageIndex) {
                elem.classList.add('active');
            }
        });
    }

    // Mark all complete when done
    if (stage === 'complete') {
        STAGE_ORDER.forEach(id => {
            const elem = document.getElementById(id);
            if (elem) {
                elem.classList.remove('active');
                elem.classList.add('completed');
            }
        });
    }
}

/**
 * Reset progress UI to initial state
 */
function resetProgress() {
    const progressFill = document.getElementById('progressFill');
    const progressPercent = document.getElementById('progressPercent');
    const progressStage = document.getElementById('progressStage');
    const loadingMessage = document.getElementById('loadingMessage');

    if (progressFill) progressFill.style.width = '0%';
    if (progressPercent) progressPercent.textContent = '0%';
    if (progressStage) progressStage.textContent = 'Starting...';
    if (loadingMessage) loadingMessage.textContent = 'Preparing analysis...';

    STAGE_ORDER.forEach(id => {
        const elem = document.getElementById(id);
        if (elem) elem.classList.remove('active', 'completed');
    });
}

/**
 * Analyze video using SSE streaming endpoint for real-time progress
 * @param {FormData} formData - Form data with video file
 * @param {Object} options - Additional options (use_tta, method, etc.)
 * @returns {Promise<Object>} - Analysis results
 */
async function analyzeWithProgress(formData, options = {}) {
    return new Promise(async (resolve, reject) => {
        let timeoutId = null;
        let abortController = null;

        // Reset timeout on each data received - 90 second inactivity timeout
        // Audio analysis with librosa can take 20-40 seconds on CPU-bound servers
        const INACTIVITY_TIMEOUT = 90000;

        const resetTimeout = () => {
            if (timeoutId) clearTimeout(timeoutId);
            timeoutId = setTimeout(() => {
                console.error('SSE inactivity timeout - no data received for 90 seconds');
                if (abortController) abortController.abort();
                reject(new Error('Connection timeout - please try again'));
            }, INACTIVITY_TIMEOUT);
        };

        try {
            // Build query params
            const params = new URLSearchParams({
                use_tta: options.use_tta ?? true,
                method: options.method ?? 'face+middle',
                generate_report: options.generate_report ?? true
            });

            abortController = new AbortController();

            const response = await fetch(`${API_BASE_URL}/stream/analyze?${params}`, {
                method: 'POST',
                body: formData,
                signal: abortController.signal
            });

            if (!response.ok) {
                const error = await response.text();
                throw new Error(error || 'Analysis failed');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            // Start the inactivity timeout
            resetTimeout();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                // Reset timeout on each chunk received
                resetTimeout();

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || ''; // Keep incomplete line in buffer

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));

                            // Update progress UI
                            updateProgress(data.stage, data.progress, data.message);

                            // Check for completion or error
                            if (data.stage === 'complete' && data.result) {
                                if (timeoutId) clearTimeout(timeoutId);
                                resolve(data.result);
                                return;
                            } else if (data.stage === 'error') {
                                if (timeoutId) clearTimeout(timeoutId);
                                reject(new Error(data.message));
                                return;
                            }
                        } catch (parseError) {
                            console.warn('Failed to parse SSE data:', line);
                        }
                    }
                }
            }

            // If we get here without a result, something went wrong
            if (timeoutId) clearTimeout(timeoutId);
            reject(new Error('Stream ended without result'));

        } catch (error) {
            if (timeoutId) clearTimeout(timeoutId);
            // Convert AbortError to a more user-friendly message
            if (error.name === 'AbortError') {
                reject(new Error('Analysis was cancelled or timed out'));
            } else {
                reject(error);
            }
        }
    });
}

let selectedFile = null;
let currentResults = null;
let radarChart = null;
let videoThumbnail = null;  // Stores video frame thumbnail for share cards

// Trait configuration with emojis and colors
const TRAIT_CONFIG = {
    openness: { emoji: '🌈', label: 'Openness', abbr: 'OPN', color: '#f59e0b' },
    conscientiousness: { emoji: '🎯', label: 'Conscientiousness', abbr: 'CNS', color: '#f59e0b' },
    extraversion: { emoji: '⚡', label: 'Extraversion', abbr: 'EXT', color: '#f59e0b' },
    agreeableness: { emoji: '🤝', label: 'Agreeableness', abbr: 'AGR', color: '#22c55e' },
    neuroticism: { emoji: '🧘', label: 'Neuroticism', abbr: 'NRT', color: '#ef4444' }
};

// The Superpower Question - broken into 3 sequential prompts
// Each prompt elicits different Big Five traits through a single creative scenario
const ASSESSMENT_QUESTIONS = [
    {
        id: 'superpower_what',
        number: 1,
        trait: 'Openness',
        question: "If you had a superpower, what would it be?",
        hint: "Imagination & novelty",
        timeLimit: 30,
        color: '#f59e0b',
        gradient: 'linear-gradient(135deg, #f59e0b 0%, #ec4899 100%)'
    },
    {
        id: 'superpower_use',
        number: 2,
        trait: 'Conscientiousness',
        question: "How would you use this superpower in your daily life?",
        hint: "Planning & habits",
        timeLimit: 30,
        color: '#3b82f6',
        gradient: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)'
    },
    {
        id: 'superpower_fails',
        number: 3,
        trait: 'Neuroticism',
        question: "What would you do when your superpower doesn't work?",
        hint: "Stress & coping",
        timeLimit: 30,
        color: '#8b5cf6',
        gradient: 'linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%)'
    }
];

// Assessment state for sequential question flow
let assessmentState = {
    isActive: false,
    currentQuestionIndex: 0,
    totalQuestions: ASSESSMENT_QUESTIONS.length,
    questionStartTimes: [],  // Track when each question started (for timestamps)
    recordedBlob: null,      // Single continuous recording
    totalDuration: 0,
    questionsAnswered: 0,
    questionsSkipped: 0
};

// Hardcoded "Room for improvement" tips based on personality profile
// These are general improvement suggestions that apply broadly
const IMPROVEMENT_TIPS = [
    { icon: "↗", text: "Explore new inputs" },
    { icon: "↗", text: "Challenge assumptions" },
    { icon: "↗", text: "Create without outcome" }
];

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

function setupEventListeners() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type by MIME type or file extension
    const validImageTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/bmp', 'image/webp'];
    const validVideoTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska', 'video/webm'];
    const validExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.webm'];

    const ext = '.' + file.name.split('.').pop().toLowerCase();
    const typeMatch = validImageTypes.includes(file.type) || validVideoTypes.includes(file.type);
    const extMatch = validExtensions.includes(ext);

    if (!typeMatch && !extMatch) {
        alert('Please upload a valid image (JPG, PNG) or video (MP4, AVI, MOV) file.');
        return;
    }

    selectedFile = file;
    showPreview(file);
}

function showPreview(file) {
    const previewSection = document.getElementById('previewSection');
    const imagePreview = document.getElementById('imagePreview');
    const videoPreview = document.getElementById('videoPreview');

    // Hide upload card and requirements (new Firasa design)
    const uploadCard = document.getElementById('uploadCard');
    const videoRequirements = document.querySelector('.video-requirements');
    const uploadPageTitle = document.querySelector('.upload-page-title');

    if (uploadCard) uploadCard.style.display = 'none';
    if (videoRequirements) videoRequirements.style.display = 'none';
    if (uploadPageTitle) uploadPageTitle.style.display = 'none';

    // Also hide old upload area just in case
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) uploadArea.style.display = 'none';

    previewSection.style.display = 'block';

    // Reset video thumbnail
    videoThumbnail = null;

    // Show preview using object URLs (faster and more reliable than data URLs)
    const objectUrl = URL.createObjectURL(file);
    const fileExt = '.' + file.name.split('.').pop().toLowerCase();
    const imageExts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp'];
    const isImage = file.type.startsWith('image/') || imageExts.includes(fileExt);

    if (isImage) {
        imagePreview.src = objectUrl;
        imagePreview.style.display = 'block';
        videoPreview.style.display = 'none';
    } else {
        videoPreview.src = objectUrl;
        videoPreview.style.display = 'block';
        imagePreview.style.display = 'none';

        // Capture video thumbnail when video is loaded
        videoPreview.onloadeddata = null; // Clear any previous handler
        videoPreview.addEventListener('loadeddata', function onPreviewLoaded() {
            videoPreview.removeEventListener('loadeddata', onPreviewLoaded);
            captureVideoThumbnail(videoPreview);
        });
    }
}

// Capture a frame from video to use as thumbnail in share cards
function captureVideoThumbnail(video) {
    try {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');

        // Seek to 1 second or 10% into the video for a better frame
        const seekTime = Math.min(1, video.duration * 0.1);
        video.currentTime = seekTime;

        // Use a one-time seeked handler to avoid interfering with later playback
        function onThumbnailSeeked() {
            video.removeEventListener('seeked', onThumbnailSeeked);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            videoThumbnail = canvas.toDataURL('image/jpeg', 0.8);
            // Reset video to start (use another one-time handler to clean up)
            function onResetSeeked() {
                video.removeEventListener('seeked', onResetSeeked);
            }
            video.addEventListener('seeked', onResetSeeked);
            video.currentTime = 0;
        }
        video.addEventListener('seeked', onThumbnailSeeked);
    } catch (e) {
        console.warn('Could not capture video thumbnail:', e);
    }
}

function resetUpload() {
    selectedFile = null;
    currentResults = null;
    videoThumbnail = null;

    // Reset UI - show upload card and requirements (new Firasa design)
    const uploadCard = document.getElementById('uploadCard');
    const videoRequirements = document.querySelector('.video-requirements');
    const uploadPageTitle = document.querySelector('.upload-page-title');

    if (uploadCard) uploadCard.style.display = 'block';
    if (videoRequirements) videoRequirements.style.display = 'block';
    if (uploadPageTitle) uploadPageTitle.style.display = '';

    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('fileInput').value = '';

    // Clear previews and revoke object URLs to free memory
    const imgPreview = document.getElementById('imagePreview');
    const vidPreview = document.getElementById('videoPreview');
    if (imgPreview.src && imgPreview.src.startsWith('blob:')) URL.revokeObjectURL(imgPreview.src);
    if (vidPreview.src && vidPreview.src.startsWith('blob:')) URL.revokeObjectURL(vidPreview.src);
    // Clear any lingering event handlers before removing src
    vidPreview.onloadeddata = null;
    vidPreview.onseeked = null;
    imgPreview.removeAttribute('src');
    vidPreview.removeAttribute('src');

    // Clear insights (use removeAttribute to avoid 404 from empty src)
    document.getElementById('insightsPhoto').removeAttribute('src');
    const insVid = document.getElementById('insightsVideo');
    insVid.removeAttribute('src');
    insVid.load();

    // Destroy chart if exists
    if (radarChart) {
        radarChart.destroy();
        radarChart = null;
    }

    // Remove results view class from body
    document.body.classList.remove('results-view');
}

// Get the best available preview image (image preview or video thumbnail)
function getPreviewImageSrc() {
    const imagePreview = document.getElementById('imagePreview');
    if (imagePreview && imagePreview.src && imagePreview.style.display !== 'none') {
        return imagePreview.src;
    }
    if (videoThumbnail) {
        return videoThumbnail;
    }
    return null;
}

async function analyzePerson() {
    if (!selectedFile) {
        alert('Please select a file first');
        return;
    }

    // Show loading and reset progress - hide everything else
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('uploadSection').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'flex';

    // Scroll to top so loading is visible
    window.scrollTo({ top: 0, behavior: 'instant' });

    resetProgress();

    // Check if file is a video
    const isVideo = selectedFile.type.startsWith('video/') ||
                    /\.(mp4|avi|mov|webm|mkv)$/i.test(selectedFile.name);

    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        let results;

        if (isVideo) {
            // Use SSE streaming endpoint for videos (real-time progress)
            results = await analyzeWithProgress(formData, {
                use_tta: true,
                method: 'face+middle',
                generate_report: true
            });
        } else {
            // Use regular endpoint for images (no progress needed)
            updateProgress('analyzing_video', 50, 'Analyzing image...');

            const response = await fetch(`${API_BASE_URL}/predict/upload?use_tta=true&method=face%2Bmiddle&include_interpretations=true&generate_report=true`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Prediction failed');
            }

            results = await response.json();
            updateProgress('complete', 100, 'Analysis complete!');
        }

        currentResults = results;

        // Hide loading, show results
        document.getElementById('loadingSection').style.display = 'none';
        displayResults(results);

    } catch (error) {
        console.error('Error:', error);
        document.getElementById('loadingSection').style.display = 'none';
        document.getElementById('previewSection').style.display = 'block';
        alert(`Analysis failed: ${error.message}`);
    }
}

function displayResults(results) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';

    // Switch to results view (white background)
    document.body.classList.add('results-view');

    // Initialize navigation
    initResultsNavigation();

    // Display debug panel first (at top)
    displayDebugPanel(results);

    // Display insights card (title, description, circular scores)
    displayInsights(results);

    // Display circular score badges in header
    displayCircularScores(results);

    // Display radar chart
    displayRadarChart(results);

    // Display trait accordion (replaces trait bars)
    displayTraitAccordion(results);

    // Display shareable cards
    displayShareableCards(results);

    // Display personality story
    displayPersonalityStory(results);

    // Display openness to experience metrics
    displayOpennessMetrics(results);

    // Display learning & growth metrics
    displayLearningMetrics(results);

    // Display relationships & empathy metrics
    displayRelationshipMetrics(results);

    // Display work DNA & focus metrics
    displayWorkMetrics(results);

    // Display creativity pulse metrics
    displayCreativityMetrics(results);

    // Display stress & resilience metrics
    displayStressMetrics(results);

    // Display voice & communication metrics
    displayAudioMetrics(results);

    // Display similarity to famous personalities section (placeholder)
    displaySimilaritySection();

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function displayInsights(results) {
    const insights = results.insights || {};
    const previewImage = document.getElementById('imagePreview');
    const previewVideo = document.getElementById('videoPreview');
    const insightsPhoto = document.getElementById('insightsPhoto');
    const insightsVideo = document.getElementById('insightsVideo');

    // Set profile photo/video
    if (selectedFile && selectedFile.type.startsWith('image/')) {
        insightsPhoto.src = previewImage.src;
        insightsPhoto.style.display = 'block';
        if (insightsVideo) insightsVideo.style.display = 'none';
    } else if (selectedFile && selectedFile.type.startsWith('video/')) {
        if (insightsVideo) {
            insightsVideo.src = previewVideo.src;
            insightsVideo.style.display = 'block';
        }
        insightsPhoto.style.display = 'none';
    } else if (videoThumbnail) {
        // Use captured thumbnail from camera recording (assessment flow)
        insightsPhoto.src = videoThumbnail;
        insightsPhoto.style.display = 'block';
        if (insightsVideo) insightsVideo.style.display = 'none';
    } else {
        // No image available - hide the photo element
        insightsPhoto.style.display = 'none';
        if (insightsVideo) insightsVideo.style.display = 'none';
    }

    // Set title
    const titleElement = document.getElementById('insightsTitle');
    titleElement.textContent = insights.title || 'Personality Profile';

    // Set description
    const descriptionElement = document.getElementById('insightsDescription');
    descriptionElement.textContent = insights.description ||
        'Your personality analysis is complete. See the detailed scores below.';
}

function displayCircularScores(results) {
    const container = document.getElementById('circularScores');
    if (!container) return;

    const interpretations = results.interpretations || {};
    const traitOrder = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'];

    container.innerHTML = traitOrder.map(trait => {
        const config = TRAIT_CONFIG[trait];
        const interp = interpretations[trait] || {};
        const percentage = Math.round(interp.t_score || 50);

        // Determine level and color
        let level, levelClass;
        if (percentage >= 65) {
            level = 'High';
            levelClass = 'high';
        } else if (percentage >= 45) {
            level = 'Moderate';
            levelClass = 'moderate';
        } else {
            level = 'Low';
            levelClass = 'low';
        }

        // Special color for neuroticism (inverted - low is good)
        const isNeuroticism = trait === 'neuroticism';

        return `
            <div class="circular-score-item ${levelClass} ${isNeuroticism ? 'neuroticism' : ''}">
                <div class="circular-score-ring">
                    <span class="circular-score-value">${percentage}%</span>
                </div>
                <span class="circular-score-level">${level}</span>
                <span class="circular-score-label">${config.label}</span>
            </div>
        `;
    }).join('');
}

function displayTraitAccordion(results) {
    const container = document.getElementById('traitAccordion');
    if (!container) return;

    const interpretations = results.interpretations || {};
    const traitOrder = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'];

    container.innerHTML = traitOrder.map((trait, index) => {
        const config = TRAIT_CONFIG[trait];
        const interp = interpretations[trait] || {};
        const percentage = Math.round(interp.t_score || 50);

        // Determine level
        let level, levelClass;
        if (percentage >= 65) {
            level = 'High';
            levelClass = 'high';
        } else if (percentage >= 45) {
            level = 'Moderate';
            levelClass = 'moderate';
        } else {
            level = 'Low';
            levelClass = 'low';
        }

        // Get description from interpretations
        const description = interp.interpretation || 'Analysis complete.';

        // First item expanded by default
        const isExpanded = index === 0;

        return `
            <div class="accordion-item ${isExpanded ? 'expanded' : ''}" data-trait="${trait}">
                <div class="accordion-header" onclick="toggleAccordion(this)">
                    <div class="accordion-title-section">
                        <span class="accordion-score ${levelClass}">${percentage}%</span>
                        <span class="accordion-trait-name">${config.label}</span>
                        <span class="accordion-level-badge ${levelClass}">${level}</span>
                    </div>
                    <svg class="accordion-chevron" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="6 9 12 15 18 9"></polyline>
                    </svg>
                </div>
                <div class="accordion-content">
                    <div class="accordion-section">
                        <p class="accordion-description">${description}</p>
                        <button class="how-to-increase-btn" onclick="openAiChatWithQuestion('${config.label}', event)">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path>
                            </svg>
                            <span>How to improve</span>
                        </button>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function toggleAccordion(header) {
    const item = header.closest('.accordion-item');
    const wasExpanded = item.classList.contains('expanded');

    // Close all other items
    document.querySelectorAll('.accordion-item').forEach(i => {
        i.classList.remove('expanded');
    });

    // Toggle this item
    if (!wasExpanded) {
        item.classList.add('expanded');
    }
}

function shareTraitResult(trait) {
    const config = TRAIT_CONFIG[trait];
    const item = document.querySelector(`.accordion-item[data-trait="${trait}"]`);
    const score = item?.querySelector('.accordion-score')?.textContent || '';
    const level = item?.querySelector('.accordion-level-badge')?.textContent || '';

    const text = `My ${config.label} score: ${score} (${level}) - Big 5 Personality Analysis`;

    if (navigator.share) {
        navigator.share({ title: 'Personality Trait', text: text });
    } else {
        navigator.clipboard.writeText(text);
        alert('Copied to clipboard!');
    }
}

function shareRadarResults() {
    const text = 'Check out my Big 5 Personality Analysis results!';
    if (navigator.share) {
        navigator.share({ title: 'Big 5 Personality', text: text });
    } else {
        navigator.clipboard.writeText(text);
        alert('Copied to clipboard!');
    }
}

function displayRadarChart(results) {
    const ctx = document.getElementById('radarChart');
    const interpretations = results.interpretations || {};

    // Trait order for radar chart (matching the image layout)
    const traitOrder = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'];
    const labels = traitOrder.map(t => TRAIT_CONFIG[t].abbr);
    const tScores = traitOrder.map(t => interpretations[t]?.t_score || 50);

    // Destroy existing chart
    if (radarChart) {
        radarChart.destroy();
    }

    // Create radar chart with styling matching the reference image
    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'T-Score',
                data: tScores,
                backgroundColor: 'rgba(254, 215, 170, 0.5)',
                borderColor: '#f59e0b',
                borderWidth: 2,
                pointBackgroundColor: '#f59e0b',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#f59e0b',
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    min: 0,
                    max: 100,
                    ticks: {
                        stepSize: 20,
                        display: false
                    },
                    pointLabels: {
                        font: {
                            size: 12,
                            weight: 'bold'
                        },
                        color: 'rgba(148, 163, 184, 0.9)'
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.25)'
                    },
                    angleLines: {
                        color: 'rgba(148, 163, 184, 0.25)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Score: ${context.parsed.r.toFixed(0)}%`;
                        }
                    }
                }
            }
        }
    });
}

function displayTraitBars(results) {
    const container = document.getElementById('traitBars');
    const interpretations = results.interpretations || {};

    // Trait order matching the reference image
    const traitOrder = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'];

    container.innerHTML = traitOrder.map(trait => {
        const config = TRAIT_CONFIG[trait];
        const interp = interpretations[trait] || {};
        const percentage = Math.round(interp.t_score || 50);

        // Determine bar color based on score
        let barColor;
        if (percentage >= 70) {
            barColor = '#22c55e'; // Green for high
        } else if (percentage >= 50) {
            barColor = '#f59e0b'; // Yellow/orange for medium
        } else {
            barColor = '#ef4444'; // Red for low
        }

        return `
            <div class="trait-bar-item">
                <div class="trait-bar-header">
                    <span class="trait-bar-emoji">${config.emoji}</span>
                    <span class="trait-bar-name">${config.label}</span>
                    <span class="trait-bar-percentage" style="color: ${barColor}">${percentage}%</span>
                </div>
                <div class="trait-bar-track">
                    <div class="trait-bar-fill" style="width: ${percentage}%; background-color: ${barColor}"></div>
                </div>
            </div>
        `;
    }).join('');
}

let shareRadarChart = null;

function displayShareableCards(results) {
    const insights = results.insights || {};
    const interpretations = results.interpretations || {};
    const previewImageSrc = getPreviewImageSrc();

    // Get title and tags
    const title = insights.title || 'Personality Profile';
    const tags = insights.tags || [];
    const tagsText = tags.map(t => t.label).join(' • ');

    // Card 1: Summary Card with full background image
    const shareCard1 = document.getElementById('shareCard1');
    if (shareCard1 && previewImageSrc) {
        shareCard1.style.backgroundImage = `url('${previewImageSrc}')`;
    }
    document.getElementById('shareCardTitle1').textContent = title;
    document.getElementById('shareCardTags1').textContent = tagsText;

    // Mini scores for Card 1
    const scoresContainer = document.getElementById('shareCardScores1');
    const traitOrder = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'];

    scoresContainer.innerHTML = traitOrder.slice(0, 5).map(trait => {
        const config = TRAIT_CONFIG[trait];
        const interp = interpretations[trait] || {};
        const percentage = Math.round(interp.t_score || 50);

        let barColor;
        if (percentage >= 70) barColor = '#22c55e';
        else if (percentage >= 50) barColor = '#f59e0b';
        else barColor = '#ef4444';

        return `
            <div class="mini-score">
                <span class="mini-score-label">${config.label}</span>
                <div class="mini-score-value">
                    <span class="mini-score-percent" style="color: ${barColor}">${percentage}%</span>
                    <div class="mini-score-bar">
                        <div class="mini-score-fill" style="width: ${percentage}%; background: ${barColor}"></div>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    // Card 2: Radar Card (light card with avatar)
    const shareCardPhoto2 = document.getElementById('shareCardPhoto2');
    if (shareCardPhoto2) {
        if (previewImageSrc) {
            shareCardPhoto2.src = previewImageSrc;
            shareCardPhoto2.style.display = 'block';
        } else {
            // Hide avatar if no image available
            shareCardPhoto2.style.display = 'none';
        }
    }
    document.getElementById('shareCardTitle2').textContent = title;
    document.getElementById('shareCardTags2').textContent = tagsText;

    // Mini radar chart for Card 2
    displayShareRadarChart(results);

    // Card 3: Comparison/Highlight Card with full background image
    const shareCard3 = document.getElementById('shareCard3');
    if (shareCard3 && previewImageSrc) {
        shareCard3.style.backgroundImage = `url('${previewImageSrc}')`;
    }

    // Find the highest scoring trait
    let highestTrait = 'openness';
    let highestScore = 0;
    traitOrder.forEach(trait => {
        const score = interpretations[trait]?.t_score || 50;
        if (score > highestScore) {
            highestScore = score;
            highestTrait = trait;
        }
    });

    document.getElementById('comparisonScore').textContent = Math.round(highestScore) + '%';
    document.getElementById('comparisonText').textContent = 'Your top trait is';
    document.getElementById('comparisonHighlight').textContent = TRAIT_CONFIG[highestTrait].label;

    // Initialize carousel (show first card)
    currentShareableCardIndex = 0;
    updateShareableCarousel();
}

function displayShareRadarChart(results) {
    const ctx = document.getElementById('shareRadarChart');
    if (!ctx) return;

    const interpretations = results.interpretations || {};
    const traitOrder = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'];
    const labels = traitOrder.map(t => TRAIT_CONFIG[t].abbr);
    const tScores = traitOrder.map(t => interpretations[t]?.t_score || 50);

    // Destroy existing chart
    if (shareRadarChart) {
        shareRadarChart.destroy();
    }

    // Create mini radar chart (dark theme)
    shareRadarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                data: tScores,
                backgroundColor: 'rgba(249, 115, 22, 0.25)',
                borderColor: '#f97316',
                borderWidth: 2,
                pointBackgroundColor: '#f97316',
                pointBorderColor: '#fff',
                pointRadius: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    min: 0,
                    max: 100,
                    ticks: { display: false },
                    pointLabels: {
                        font: { size: 9 },
                        color: '#94a3b8'
                    },
                    grid: { color: 'rgba(148, 163, 184, 0.15)' },
                    angleLines: { color: 'rgba(148, 163, 184, 0.15)' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function displayPersonalityStory(results) {
    const storySection = document.getElementById('personalityStorySection');
    if (!storySection) return;

    const insights = results.insights || {};

    // Check if we have story content from the LLM
    if (!insights.quote && !insights.story) {
        // Hide section if no story content
        storySection.style.display = 'none';
        return;
    }

    storySection.style.display = 'block';

    // Set quote (without extra quotes since wireframe doesn't show them)
    const storyQuote = document.getElementById('storyQuote');
    if (storyQuote) {
        storyQuote.textContent = insights.quote || 'Your unique personality shines through in everything you do.';
    }

    // Set narrative/story (as HTML paragraphs)
    const storyNarrative = document.getElementById('storyNarrative');
    if (storyNarrative && insights.story) {
        // Split story into paragraphs if it's long
        const paragraphs = insights.story.split('\n\n').filter(p => p.trim());
        if (paragraphs.length > 1) {
            storyNarrative.innerHTML = paragraphs.map(p => `<p>${p}</p>`).join('');
        } else {
            storyNarrative.innerHTML = `<p>${insights.story}</p>`;
        }
    }

    // Set story traits ("You are:" tags)
    const storyTraitsGrid = document.getElementById('storyTraitsGrid');
    if (storyTraitsGrid && insights.story_traits && insights.story_traits.length > 0) {
        storyTraitsGrid.innerHTML = insights.story_traits.map(tag => `
            <span class="story-trait-tag-new">
                <span class="tag-emoji">${tag.emoji}</span>
                <span class="tag-label">${tag.label}</span>
            </span>
        `).join('');
    }

    // Set improvement tips ("Room for improvement:" section)
    const improvementGrid = document.getElementById('storyImprovementGrid');
    if (improvementGrid) {
        improvementGrid.innerHTML = IMPROVEMENT_TIPS.map(tip => `
            <span class="story-improvement-tag">
                <span class="improvement-icon">${tip.icon}</span>
                <span class="improvement-text">${tip.text}</span>
            </span>
        `).join('');
    }
}

function sharePersonalityStory() {
    const quote = document.getElementById('storyQuote')?.textContent || '';
    const text = `"${quote}" - My unique personality story from Big 5 Analysis`;

    if (navigator.share) {
        navigator.share({ title: 'My Personality Story', text: text });
    } else {
        navigator.clipboard.writeText(text);
        alert('Copied to clipboard!');
    }
}

// Metric display names and icons
const RELATIONSHIP_METRIC_CONFIG = {
    trust_signaling: { label: 'Trust Signaling', icon: '🤝' },
    social_openness: { label: 'Social Openness', icon: '🌐' },
    empathic_disposition: { label: 'Empathic Disposition', icon: '💝' },
    conflict_avoidance: { label: 'Conflict Avoidance', icon: '🕊️' },
    harmony_seeking: { label: 'Harmony Seeking', icon: '☮️' },
    anxiety_avoidance: { label: 'Anxiety Avoidance', icon: '🧘' }
};

// Helper function to generate section accordion HTML
function generateSectionAccordion(metricsData, sectionId) {
    const accordionItems = [];

    // Snapshot Insight
    if (metricsData.snapshot_insight) {
        accordionItems.push({
            id: 'snapshot',
            title: 'Snapshot insight',
            content: `<p class="accordion-text">${metricsData.snapshot_insight}</p>`
        });
    }

    // Behavioral Patterns - Image cards layout
    if (metricsData.behavioral_patterns && metricsData.behavioral_patterns.length > 0) {
        // Use captured video thumbnail if available
        const thumbnailSrc = videoThumbnail || '';
        const imageStyle = thumbnailSrc ? '' : 'style="background: linear-gradient(135deg, #f0f0f0 0%, #e0e0e0 100%);"';

        const patternsHtml = metricsData.behavioral_patterns.map(p => `
            <div class="insight-image-card">
                <div class="insight-card-image" ${!thumbnailSrc ? imageStyle : ''}>
                    ${thumbnailSrc ? `<img src="${thumbnailSrc}" alt="Pattern">` : '<div class="insight-image-placeholder"></div>'}
                    <div class="insight-card-actions">
                        <button class="insight-action-btn" title="Capture">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                                <circle cx="8.5" cy="8.5" r="1.5"></circle>
                                <polyline points="21 15 16 10 5 21"></polyline>
                            </svg>
                        </button>
                        <button class="insight-action-btn" title="Share">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="18" cy="5" r="3"></circle>
                                <circle cx="6" cy="12" r="3"></circle>
                                <circle cx="18" cy="19" r="3"></circle>
                                <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
                                <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="insight-card-content">
                    <div class="insight-card-icon pattern-icon"></div>
                    <span class="insight-card-title">${p.title}</span>
                    <span class="insight-card-desc">${p.description}</span>
                </div>
            </div>
        `).join('');
        accordionItems.push({
            id: 'patterns',
            title: 'Behavioral patterns observed',
            content: `<div class="insight-cards-grid">${patternsHtml}</div>`
        });
    }

    // How Others Experience You
    if (metricsData.how_others_experience) {
        accordionItems.push({
            id: 'others',
            title: 'How others experience you',
            content: `<p class="accordion-text">${metricsData.how_others_experience}</p>`
        });
    }

    // Strength & Trade-off - Image cards layout
    if (metricsData.strength || metricsData.tradeoff) {
        // Use captured video thumbnail if available
        const thumbnailSrc = videoThumbnail || '';
        const imageStyle = thumbnailSrc ? '' : 'style="background: linear-gradient(135deg, #f0f0f0 0%, #e0e0e0 100%);"';

        let strengthHtml = '';
        if (metricsData.strength) {
            strengthHtml += `
                <div class="insight-image-card strength-card">
                    <div class="insight-card-image" ${!thumbnailSrc ? imageStyle : ''}>
                        ${thumbnailSrc ? `<img src="${thumbnailSrc}" alt="Strength">` : '<div class="insight-image-placeholder"></div>'}
                        <div class="insight-card-actions">
                            <button class="insight-action-btn" title="Capture">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                                    <circle cx="8.5" cy="8.5" r="1.5"></circle>
                                    <polyline points="21 15 16 10 5 21"></polyline>
                                </svg>
                            </button>
                            <button class="insight-action-btn" title="Share">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <circle cx="18" cy="5" r="3"></circle>
                                    <circle cx="6" cy="12" r="3"></circle>
                                    <circle cx="18" cy="19" r="3"></circle>
                                    <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
                                    <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
                                </svg>
                            </button>
                        </div>
                    </div>
                    <div class="insight-card-content">
                        <div class="insight-card-icon strength-icon"></div>
                        <span class="insight-card-title">Strength</span>
                        <span class="insight-card-desc">${metricsData.strength.title || metricsData.strength.description}</span>
                    </div>
                </div>
            `;
        }
        if (metricsData.tradeoff) {
            strengthHtml += `
                <div class="insight-image-card tradeoff-card">
                    <div class="insight-card-image" ${!thumbnailSrc ? imageStyle : ''}>
                        ${thumbnailSrc ? `<img src="${thumbnailSrc}" alt="Trade-Off">` : '<div class="insight-image-placeholder"></div>'}
                        <div class="insight-card-actions">
                            <button class="insight-action-btn" title="Capture">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                                    <circle cx="8.5" cy="8.5" r="1.5"></circle>
                                    <polyline points="21 15 16 10 5 21"></polyline>
                                </svg>
                            </button>
                            <button class="insight-action-btn" title="Share">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <circle cx="18" cy="5" r="3"></circle>
                                    <circle cx="6" cy="12" r="3"></circle>
                                    <circle cx="18" cy="19" r="3"></circle>
                                    <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
                                    <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
                                </svg>
                            </button>
                        </div>
                    </div>
                    <div class="insight-card-content">
                        <div class="insight-card-icon tradeoff-icon"></div>
                        <span class="insight-card-title">Trade-Off</span>
                        <span class="insight-card-desc">${metricsData.tradeoff.title || metricsData.tradeoff.description}</span>
                    </div>
                </div>
            `;
        }
        accordionItems.push({
            id: 'strength',
            title: 'Strength & trade-off',
            content: `<div class="insight-cards-grid">${strengthHtml}</div>`
        });
    }

    // Growth Lever
    if (metricsData.growth_lever) {
        accordionItems.push({
            id: 'growth',
            title: 'Growth lever',
            content: `<p class="accordion-text">${metricsData.growth_lever}</p>`
        });
    }

    // Coach Recommendation
    if (metricsData.coach_recommendation) {
        let coachHtml = `<p class="accordion-text coach-rec">${metricsData.coach_recommendation}</p>`;
        if (metricsData.actionable_steps && metricsData.actionable_steps.length > 0) {
            const stepsHtml = metricsData.actionable_steps.map(step => `
                <span class="action-step-tag">
                    <span class="step-emoji">${step.emoji}</span>
                    <span class="step-text">${step.text}</span>
                </span>
            `).join('');
            coachHtml += `<div class="action-steps-grid">${stepsHtml}</div>`;
        }
        accordionItems.push({
            id: 'coach',
            title: 'Coach recommendation',
            content: coachHtml
        });
    }

    // Generate accordion HTML
    return accordionItems.map((item, index) => `
        <div class="section-accordion-item ${index === 0 ? 'expanded' : ''}" data-item="${item.id}">
            <div class="section-accordion-header" onclick="toggleSectionAccordion(this, '${sectionId}')">
                <span class="section-accordion-title">${item.title}</span>
                <svg class="section-accordion-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
            </div>
            <div class="section-accordion-content">
                ${item.content}
            </div>
        </div>
    `).join('');
}

// Toggle section accordion items
function toggleSectionAccordion(header, sectionId) {
    const item = header.closest('.section-accordion-item');
    const accordion = document.getElementById(sectionId);
    const wasExpanded = item.classList.contains('expanded');

    // Close all items in this accordion
    accordion.querySelectorAll('.section-accordion-item').forEach(i => {
        i.classList.remove('expanded');
    });

    // Toggle this item
    if (!wasExpanded) {
        item.classList.add('expanded');
    }
}

// Helper to generate gauge circles for metrics
function generateGaugeCircles(metrics, metricConfig, metricOrder) {
    return metricOrder
        .filter(key => metrics[key])
        .map(key => {
            const metric = metrics[key];
            const config = metricConfig[key] || { label: key };
            const score = Math.round(metric.score || 0);
            const level = metric.level || 'Moderate';

            let levelClass = 'moderate';
            if (level === 'High') levelClass = 'high';
            else if (level === 'Low') levelClass = 'low';

            return `
                <div class="gauge-circle-item ${levelClass}">
                    <div class="gauge-circle">
                        <span class="gauge-value">${score}%</span>
                    </div>
                    <span class="gauge-level-badge ${levelClass}">${level}</span>
                    <span class="gauge-label">${config.label}</span>
                </div>
            `;
        }).join('');
}

// Helper to generate suitable for tags
function generateSuitableForTags(tags) {
    if (!tags || tags.length === 0) return '';
    return tags.map(tag => `<span class="suitable-tag">${tag}</span>`).join('');
}

function displayRelationshipMetrics(results) {
    const relationshipsSection = document.getElementById('relationshipsSection');
    if (!relationshipsSection) return;

    const relationshipMetrics = results.relationship_metrics;

    // Hide section if no relationship metrics data
    if (!relationshipMetrics || !relationshipMetrics.metrics) {
        relationshipsSection.style.display = 'none';
        return;
    }

    relationshipsSection.style.display = 'block';

    const metrics = relationshipMetrics.metrics;

    // Display big score (empathic_disposition as primary)
    const bigScoreEl = document.getElementById('relBigScore');
    if (bigScoreEl && metrics.empathic_disposition) {
        const score = Math.round(metrics.empathic_disposition.score || 0);
        bigScoreEl.querySelector('.big-score-value').textContent = `${score}%`;
        // Set color based on level
        const level = metrics.empathic_disposition.level || 'Moderate';
        bigScoreEl.classList.remove('high', 'moderate', 'low');
        bigScoreEl.classList.add(level.toLowerCase());
    }

    // Display gauge circles (other metrics)
    const gaugesEl = document.getElementById('relationshipGauges');
    if (gaugesEl) {
        const gaugeOrder = ['trust_signaling', 'social_openness', 'conflict_avoidance'];
        gaugesEl.innerHTML = generateGaugeCircles(metrics, RELATIONSHIP_METRIC_CONFIG, gaugeOrder);
    }

    // Display suitable for tags
    const suitableEl = document.getElementById('relSuitableForTags');
    if (suitableEl) {
        suitableEl.innerHTML = generateSuitableForTags(relationshipMetrics.suitable_for);
    }

    // Display accordion
    const accordionEl = document.getElementById('relationshipAccordion');
    if (accordionEl) {
        accordionEl.innerHTML = generateSectionAccordion(relationshipMetrics, 'relationshipAccordion');
    }
}

// Work metrics display names and icons
const WORK_METRIC_CONFIG = {
    persistence: { label: 'Persistence', icon: '🎯' },
    focus_attention: { label: 'Focus & Attention', icon: '🧠' },
    structure_preference: { label: 'Structure Preference', icon: '📋' },
    risk_aversion: { label: 'Risk Aversion', icon: '🛡️' }
};

function displayWorkMetrics(results) {
    const workSection = document.getElementById('workSection');
    if (!workSection) return;

    const workMetrics = results.work_metrics;

    // Hide section if no work metrics data
    if (!workMetrics || !workMetrics.metrics) {
        workSection.style.display = 'none';
        return;
    }

    workSection.style.display = 'block';

    const metrics = workMetrics.metrics;

    // Display big score (persistence as primary)
    const bigScoreEl = document.getElementById('workBigScore');
    if (bigScoreEl && metrics.persistence) {
        const score = Math.round(metrics.persistence.score || 0);
        bigScoreEl.querySelector('.big-score-value').textContent = `${score}%`;
        const level = metrics.persistence.level || 'Moderate';
        bigScoreEl.classList.remove('high', 'moderate', 'low');
        bigScoreEl.classList.add(level.toLowerCase());
    }

    // Display gauge circles (other metrics)
    const gaugesEl = document.getElementById('workGauges');
    if (gaugesEl) {
        const gaugeOrder = ['focus_attention', 'structure_preference', 'risk_aversion'];
        gaugesEl.innerHTML = generateGaugeCircles(metrics, WORK_METRIC_CONFIG, gaugeOrder);
    }

    // Display suitable for tags
    const suitableEl = document.getElementById('workSuitableForTags');
    if (suitableEl) {
        suitableEl.innerHTML = generateSuitableForTags(workMetrics.suitable_for);
    }

    // Display accordion
    const accordionEl = document.getElementById('workAccordion');
    if (accordionEl) {
        accordionEl.innerHTML = generateSectionAccordion(workMetrics, 'workAccordion');
    }
}

// Creativity metrics display names and icons
const CREATIVITY_METRIC_CONFIG = {
    ideation_power: { label: 'Ideation Power', icon: '💡' },
    openness_to_novelty: { label: 'Openness to Novelty', icon: '🌱' },
    originality_index: { label: 'Originality Index', icon: '✨' },
    attention_to_detail_creative: { label: 'Attention to Detail', icon: '🔍' }
};

function displayCreativityMetrics(results) {
    const creativitySection = document.getElementById('creativitySection');
    if (!creativitySection) return;

    const creativityMetrics = results.creativity_metrics;

    // Hide section if no creativity metrics data
    if (!creativityMetrics || !creativityMetrics.metrics) {
        creativitySection.style.display = 'none';
        return;
    }

    creativitySection.style.display = 'block';

    const metrics = creativityMetrics.metrics;

    // Display big score (ideation_power as primary)
    const bigScoreEl = document.getElementById('creativityBigScore');
    if (bigScoreEl && metrics.ideation_power) {
        const score = Math.round(metrics.ideation_power.score || 0);
        bigScoreEl.querySelector('.big-score-value').textContent = `${score}%`;
        const level = metrics.ideation_power.level || 'Moderate';
        bigScoreEl.classList.remove('high', 'moderate', 'low');
        bigScoreEl.classList.add(level.toLowerCase());
    }

    // Display gauge circles (other metrics)
    const gaugesEl = document.getElementById('creativityGauges');
    if (gaugesEl) {
        const gaugeOrder = ['openness_to_novelty', 'originality_index', 'attention_to_detail_creative'];
        gaugesEl.innerHTML = generateGaugeCircles(metrics, CREATIVITY_METRIC_CONFIG, gaugeOrder);
    }

    // Display suitable for tags
    const suitableEl = document.getElementById('creativitySuitableForTags');
    if (suitableEl) {
        suitableEl.innerHTML = generateSuitableForTags(creativityMetrics.suitable_for);
    }

    // Display accordion
    const accordionEl = document.getElementById('creativityAccordion');
    if (accordionEl) {
        accordionEl.innerHTML = generateSectionAccordion(creativityMetrics, 'creativityAccordion');
    }
}

// Stress metrics display names and icons
const STRESS_METRIC_CONFIG = {
    stress_indicators: { label: 'Stress Indicators', icon: '⚡' },
    emotional_regulation: { label: 'Emotional Regulation', icon: '🧘' },
    resilience_score: { label: 'Resilience Score', icon: '💪' }
};

function displayStressMetrics(results) {
    const stressSection = document.getElementById('stressSection');
    if (!stressSection) return;

    const stressMetrics = results.stress_metrics;

    // Hide section if no stress metrics data
    if (!stressMetrics || !stressMetrics.metrics) {
        stressSection.style.display = 'none';
        return;
    }

    stressSection.style.display = 'block';

    const metrics = stressMetrics.metrics;

    // Display big score (resilience_score as primary)
    const bigScoreEl = document.getElementById('stressBigScore');
    if (bigScoreEl && metrics.resilience_score) {
        const score = Math.round(metrics.resilience_score.score || 0);
        bigScoreEl.querySelector('.big-score-value').textContent = `${score}%`;
        const level = metrics.resilience_score.level || 'Moderate';
        bigScoreEl.classList.remove('high', 'moderate', 'low');
        bigScoreEl.classList.add(level.toLowerCase());
    }

    // Display gauge circles (other metrics)
    const gaugesEl = document.getElementById('stressGauges');
    if (gaugesEl) {
        const gaugeOrder = ['emotional_regulation', 'stress_indicators'];
        gaugesEl.innerHTML = generateGaugeCircles(metrics, STRESS_METRIC_CONFIG, gaugeOrder);
    }

    // Display suitable for tags
    const suitableEl = document.getElementById('stressSuitableForTags');
    if (suitableEl) {
        suitableEl.innerHTML = generateSuitableForTags(stressMetrics.suitable_for);
    }

    // Display accordion
    const accordionEl = document.getElementById('stressAccordion');
    if (accordionEl) {
        accordionEl.innerHTML = generateSectionAccordion(stressMetrics, 'stressAccordion');
    }
}

// Openness metrics display names and icons
const OPENNESS_METRIC_CONFIG = {
    openness_to_experience: { label: 'Openness to Experience', icon: '🌈' },
    novelty_seeking: { label: 'Novelty Seeking', icon: '🔍' },
    risk_tolerance_adventure: { label: 'Risk Tolerance (Adventure)', icon: '🎢' },
    planning_preference: { label: 'Planning Preference', icon: '📋' }
};

// Learning & Growth metric display names and icons
const LEARNING_METRIC_CONFIG = {
    intellectual_curiosity: { label: 'Intellectual Curiosity', icon: '🧠' },
    reflective_tendency: { label: 'Reflective Tendency', icon: '💭' },
    structured_learning_preference: { label: 'Structured Learning Preference', icon: '📚' },
    adaptability_index: { label: 'Adaptability Index', icon: '🔄' }
};

// Default "Experiences you enjoy most" tags based on openness profile
function getOpennessExperienceTags(metrics) {
    const tags = [];
    const openness = metrics.openness_to_experience?.score || 50;
    const novelty = metrics.novelty_seeking?.score || 50;
    const risk = metrics.risk_tolerance_adventure?.score || 50;

    if (openness >= 60) {
        tags.push({ emoji: '🎨', text: 'Cultural Exploration' });
        tags.push({ emoji: '📚', text: 'Learning-Driven Travel' });
    }
    if (novelty >= 60) {
        tags.push({ emoji: '🌍', text: 'New Experiences' });
    }
    if (risk >= 60) {
        tags.push({ emoji: '🏔️', text: 'Adventure Activities' });
    }
    if (openness >= 50) {
        tags.push({ emoji: '💡', text: 'Idea-Focused Communities' });
    }

    // Fallback tags
    if (tags.length < 3) {
        tags.push({ emoji: '🎯', text: 'Structured Learning' });
        tags.push({ emoji: '🏠', text: 'Familiar Environments' });
    }

    return tags.slice(0, 4);
}

function displayOpennessMetrics(results) {
    const opennessSection = document.getElementById('opennessSection');
    if (!opennessSection) return;

    const opennessMetrics = results.openness_metrics;

    // Hide section if no openness metrics data
    if (!opennessMetrics || !opennessMetrics.metrics) {
        opennessSection.style.display = 'none';
        return;
    }

    opennessSection.style.display = 'block';

    const metrics = opennessMetrics.metrics;

    // Display big score (openness_to_experience as primary)
    const bigScoreEl = document.getElementById('opennessBigScore');
    if (bigScoreEl && metrics.openness_to_experience) {
        const score = Math.round(metrics.openness_to_experience.score || 0);
        const valueEl = bigScoreEl.querySelector('.big-score-value');
        if (valueEl) valueEl.textContent = `${score}%`;
        const level = metrics.openness_to_experience.level || 'Moderate';
        bigScoreEl.classList.remove('high', 'moderate', 'low');
        bigScoreEl.classList.add(level.toLowerCase());
    }

    // Display gauge circles (other metrics)
    const gaugesEl = document.getElementById('opennessGauges');
    if (gaugesEl) {
        const gaugeOrder = ['openness_to_experience', 'novelty_seeking', 'risk_tolerance_adventure'];
        gaugesEl.innerHTML = generateGaugeCircles(metrics, OPENNESS_METRIC_CONFIG, gaugeOrder);
    }

    // Display suitable for tags (experiences you enjoy most)
    const suitableEl = document.getElementById('opennessSuitableForTags');
    if (suitableEl) {
        // Use API tags if provided, otherwise generate based on metrics
        const tags = opennessMetrics.suitable_for || getOpennessExperienceTags(metrics).map(t => `${t.emoji} ${t.text}`);
        suitableEl.innerHTML = generateSuitableForTags(tags);
    }

    // Display accordion
    const accordionEl = document.getElementById('opennessAccordion');
    if (accordionEl) {
        accordionEl.innerHTML = generateSectionAccordion(opennessMetrics, 'opennessAccordion');
    }
}

// Default "Learning styles that fit you" tags based on learning profile
function getLearningStyleTags(metrics) {
    const tags = [];
    const curiosity = metrics.intellectual_curiosity?.score || 50;
    const reflective = metrics.reflective_tendency?.score || 50;
    const structured = metrics.structured_learning_preference?.score || 50;
    const adaptability = metrics.adaptability_index?.score || 50;

    if (curiosity >= 60) {
        tags.push({ emoji: '🔬', text: 'Exploratory Learning' });
        tags.push({ emoji: '📖', text: 'Self-Directed Study' });
    }
    if (reflective >= 60) {
        tags.push({ emoji: '💭', text: 'Reflective Learning' });
        tags.push({ emoji: '✍️', text: 'Journaling & Review' });
    }
    if (structured >= 60) {
        tags.push({ emoji: '📋', text: 'Structured Courses' });
        tags.push({ emoji: '🎯', text: 'Goal-Oriented Learning' });
    }
    if (adaptability >= 60) {
        tags.push({ emoji: '🔄', text: 'Adaptive Learning' });
        tags.push({ emoji: '🌱', text: 'Growth Mindset' });
    }

    // Fallback tags
    if (tags.length < 3) {
        tags.push({ emoji: '🤝', text: 'Mentorship-Driven' });
        tags.push({ emoji: '🛠️', text: 'Project-Based Growth' });
    }

    return tags.slice(0, 4);
}

function displayLearningMetrics(results) {
    const learningSection = document.getElementById('learningSection');
    if (!learningSection) return;

    const learningMetrics = results.learning_metrics;

    // Hide section if no learning metrics data
    if (!learningMetrics || !learningMetrics.metrics) {
        learningSection.style.display = 'none';
        return;
    }

    learningSection.style.display = 'block';

    const metrics = learningMetrics.metrics;

    // Display big score (intellectual_curiosity as primary)
    const bigScoreEl = document.getElementById('learningBigScore');
    if (bigScoreEl && metrics.intellectual_curiosity) {
        const score = Math.round(metrics.intellectual_curiosity.score || 0);
        const valueEl = bigScoreEl.querySelector('.big-score-value');
        if (valueEl) valueEl.textContent = `${score}%`;
        const level = metrics.intellectual_curiosity.level || 'Moderate';
        bigScoreEl.classList.remove('high', 'moderate', 'low');
        bigScoreEl.classList.add(level.toLowerCase());
    }

    // Display gauge circles (all metrics)
    const gaugesEl = document.getElementById('learningGauges');
    if (gaugesEl) {
        const gaugeOrder = ['intellectual_curiosity', 'reflective_tendency', 'adaptability_index'];
        gaugesEl.innerHTML = generateGaugeCircles(metrics, LEARNING_METRIC_CONFIG, gaugeOrder);
    }

    // Display suitable for tags (learning styles that fit you)
    const suitableEl = document.getElementById('learningSuitableForTags');
    if (suitableEl) {
        // Use API tags if provided, otherwise generate based on metrics
        const tags = learningMetrics.suitable_for || getLearningStyleTags(metrics).map(t => `${t.emoji} ${t.text}`);
        suitableEl.innerHTML = generateSuitableForTags(tags);
    }

    // Display accordion
    const accordionEl = document.getElementById('learningAccordion');
    if (accordionEl) {
        accordionEl.innerHTML = generateSectionAccordion(learningMetrics, 'learningAccordion');
    }
}

// Audio/Voice metrics display names and icons
const AUDIO_METRIC_CONFIG = {
    vocal_extraversion: { label: 'Vocal Extraversion', icon: '🎤' },
    vocal_expressiveness: { label: 'Vocal Expressiveness', icon: '🎭' },
    vocal_fluency: { label: 'Vocal Fluency', icon: '💬' }
};

// Voice characteristic icons
const VOICE_CHAR_ICONS = {
    pitch: '🎵',
    expressiveness: '🎭',
    volume: '🔊',
    pace: '⏱️',
    brightness: '✨',
    stability: '🧘'
};

function displayAudioMetrics(results) {
    const audioSection = document.getElementById('audioSection');
    if (!audioSection) return;

    const audioMetrics = results.audio_metrics;

    // Debug logging
    console.log('=== VOICE SECTION DEBUG ===');
    console.log('audioMetrics:', audioMetrics);
    console.log('coach_recommendation:', audioMetrics?.coach_recommendation);
    console.log('snapshot_insight:', audioMetrics?.snapshot_insight);
    console.log('behavioral_patterns:', audioMetrics?.behavioral_patterns);
    console.log('strength:', audioMetrics?.strength);
    console.log('tradeoff:', audioMetrics?.tradeoff);
    console.log('growth_lever:', audioMetrics?.growth_lever);
    console.log('suitable_for:', audioMetrics?.suitable_for);
    console.log('=== END DEBUG ===');

    // Hide section if no audio metrics data AND no LLM coaching
    const hasIndicators = audioMetrics && audioMetrics.indicators && Object.keys(audioMetrics.indicators).length > 0;
    const hasLLMCoaching = audioMetrics && (audioMetrics.coach_recommendation || audioMetrics.snapshot_insight);

    if (!hasIndicators && !hasLLMCoaching) {
        audioSection.style.display = 'none';
        return;
    }

    audioSection.style.display = 'block';

    // Calculate overall voice score from indicators (or use 50 as default)
    const indicators = audioMetrics.indicators || {};
    const indicatorOrder = ['vocal_extraversion', 'vocal_expressiveness', 'vocal_fluency'];
    let totalScore = 0;
    let count = 0;
    indicatorOrder.forEach(key => {
        if (indicators[key] && indicators[key].score) {
            totalScore += indicators[key].score;
            count++;
        }
    });
    const overallScore = count > 0 ? Math.round(totalScore / count) : 50;

    // Display big score
    const bigScoreContainer = document.getElementById('voiceBigScore');
    if (bigScoreContainer) {
        bigScoreContainer.innerHTML = `
            <div class="big-score-display">
                <span class="big-score-arrow">▲</span>
                <span class="big-score-value">${overallScore}%</span>
            </div>
            <span class="big-score-subtext">Vocal communication score</span>
        `;
    }

    // Display gauge circles for voice indicators
    const gaugesGrid = document.getElementById('voiceGaugesGrid');
    if (gaugesGrid) {
        gaugesGrid.innerHTML = indicatorOrder
            .filter(key => indicators[key])
            .map(key => {
                const indicator = indicators[key];
                const config = AUDIO_METRIC_CONFIG[key] || { label: key, icon: '🎤' };
                const score = Math.round(indicator.score || 0);
                const level = indicator.level || 'Moderate';
                let levelClass = 'moderate';
                if (level === 'High') levelClass = 'high';
                else if (level === 'Low') levelClass = 'low';

                return `
                    <div class="gauge-circle-item ${levelClass}">
                        <div class="gauge-circle">
                            <span class="gauge-value">${score}%</span>
                        </div>
                        <span class="gauge-level-badge ${levelClass}">${level}</span>
                        <span class="gauge-label">${config.label}</span>
                    </div>
                `;
            }).join('');
    }

    // Display suitable for tags
    const suitableForTags = document.getElementById('voiceSuitableForTags');
    if (suitableForTags && audioMetrics.suitable_for) {
        suitableForTags.innerHTML = audioMetrics.suitable_for.map(tag =>
            `<span class="suitable-tag">${tag}</span>`
        ).join('');
    } else if (suitableForTags) {
        // Default tags if not provided by LLM
        suitableForTags.innerHTML = `
            <span class="suitable-tag">Conversations</span>
            <span class="suitable-tag">Presentations</span>
            <span class="suitable-tag">Meetings</span>
        `;
    }

    // Display accordion with voice coaching content
    const accordion = document.getElementById('voiceAccordion');
    if (accordion) {
        const accordionItems = [];

        // Snapshot insight
        accordionItems.push({
            title: 'Snapshot insight',
            content: audioMetrics.snapshot_insight || audioMetrics.coach_recommendation || 'Your voice projects energy and expressiveness.',
            expanded: true
        });

        // Behavioral patterns
        if (audioMetrics.behavioral_patterns && audioMetrics.behavioral_patterns.length > 0) {
            const patternsHtml = audioMetrics.behavioral_patterns.map(p =>
                `<div class="behavioral-pattern-item">
                    <span class="pattern-title">${p.title}</span>
                    <span class="pattern-desc">${p.description}</span>
                </div>`
            ).join('');
            accordionItems.push({
                title: 'Behavioral patterns observed',
                content: patternsHtml,
                isHtml: true
            });
        }

        // How others experience
        if (audioMetrics.how_others_experience) {
            accordionItems.push({
                title: 'How others likely experience you',
                content: audioMetrics.how_others_experience
            });
        }

        // Strength & Tradeoff
        if (audioMetrics.strength || audioMetrics.tradeoff) {
            let strengthHtml = '';
            if (audioMetrics.strength) {
                strengthHtml += `<div class="strength-item">
                    <span class="strength-label">Strength:</span>
                    <span class="strength-title">${audioMetrics.strength.title}</span>
                    <p class="strength-desc">${audioMetrics.strength.description}</p>
                </div>`;
            }
            if (audioMetrics.tradeoff) {
                strengthHtml += `<div class="tradeoff-item">
                    <span class="tradeoff-label">Trade-off:</span>
                    <span class="tradeoff-title">${audioMetrics.tradeoff.title}</span>
                    <p class="tradeoff-desc">${audioMetrics.tradeoff.description}</p>
                </div>`;
            }
            accordionItems.push({
                title: 'Strength & trade-off',
                content: strengthHtml,
                isHtml: true
            });
        }

        // Growth lever
        if (audioMetrics.growth_lever) {
            accordionItems.push({
                title: 'Growth lever',
                content: audioMetrics.growth_lever
            });
        }

        // Coach recommendation (if different from snapshot)
        if (audioMetrics.coach_recommendation && audioMetrics.coach_recommendation !== audioMetrics.snapshot_insight) {
            // Just show the coach recommendation text, no actionable steps grid
            accordionItems.push({
                title: 'Coach recommendation',
                content: audioMetrics.coach_recommendation
            });
        }

        // Render accordion
        accordion.innerHTML = accordionItems.map((item, index) => `
            <div class="section-accordion-item ${item.expanded ? 'expanded' : ''}">
                <div class="section-accordion-header" onclick="toggleSectionAccordion(this, 'voiceAccordion')">
                    <span class="section-accordion-title">${item.title}</span>
                    <svg class="section-accordion-chevron" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="6 9 12 15 18 9"></polyline>
                    </svg>
                </div>
                <div class="section-accordion-content">
                    ${item.isHtml ? item.content : `<p class="accordion-text">${item.content}</p>`}
                </div>
            </div>
        `).join('');
    }
}

function shareVoiceMetrics() {
    const text = `My Voice & Communication Analysis - OCEAN Personality`;
    if (navigator.share) {
        navigator.share({ title: 'Voice Analysis', text: text });
    } else {
        navigator.clipboard.writeText(text);
        alert('Copied to clipboard!');
    }
}

// ============================================
// Similarity to Famous Personalities Section
// ============================================

/**
 * Display the similarity to famous personalities section (placeholder)
 * This shows placeholder cards that will be populated with actual data later
 */
function displaySimilaritySection() {
    const similaritySection = document.getElementById('similaritySection');
    if (!similaritySection) return;

    // Show the section (it's a placeholder for now)
    similaritySection.style.display = 'block';

    // Initialize navigation buttons (placeholder functionality)
    const prevBtn = document.getElementById('similarityPrev');
    const nextBtn = document.getElementById('similarityNext');

    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            // Placeholder: scroll left or show previous set of cards
            console.log('Previous similarity cards');
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            // Placeholder: scroll right or show next set of cards
            console.log('Next similarity cards');
        });
    }
}

// ============================================
// Debug Visualization Panel
// ============================================

let debugPanelVisible = true;

function toggleDebugPanel() {
    const debugPanel = document.getElementById('debugPanel');
    const toggleBtn = document.getElementById('debugToggleBtn');
    const content = debugPanel.querySelectorAll('.debug-section');

    debugPanelVisible = !debugPanelVisible;

    content.forEach(section => {
        section.style.display = debugPanelVisible ? 'block' : 'none';
    });

    toggleBtn.textContent = debugPanelVisible ? 'Hide' : 'Show';
}

function displayDebugPanel(results) {
    const debugPanel = document.getElementById('debugPanel');
    if (!debugPanel) return;

    const metadata = results.metadata || {};
    const debugData = metadata.debug_visualization;

    // Hide panel if no debug data
    if (!debugData) {
        debugPanel.style.display = 'none';
        return;
    }

    debugPanel.style.display = 'block';

    // Display frames with face detection boxes
    displayDebugFrames(debugData.frames || []);

    // Display transcript
    displayDebugTranscript(debugData.transcript, debugData.transcript_length);

    // Display waveform
    displayDebugWaveform(debugData.waveform);
}

function displayDebugFrames(frames) {
    const grid = document.getElementById('debugFramesGrid');
    if (!grid || frames.length === 0) return;

    grid.innerHTML = frames.map((frame, idx) => {
        const faceIndicator = frame.face_detected
            ? `<span class="face-detected">Face detected</span>`
            : `<span class="face-not-detected">No face</span>`;

        // Create face box overlay if detected
        let faceBoxStyle = '';
        if (frame.face_detected && frame.face_bbox) {
            const bbox = frame.face_bbox;
            // Calculate percentages for overlay positioning
            const left = (bbox.x / frame.width) * 100;
            const top = (bbox.y / frame.height) * 100;
            const width = (bbox.width / frame.width) * 100;
            const height = (bbox.height / frame.height) * 100;
            faceBoxStyle = `left: ${left}%; top: ${top}%; width: ${width}%; height: ${height}%;`;
        }

        return `
            <div class="debug-frame-item">
                <div class="debug-frame-container">
                    <img src="${frame.image_base64}" alt="Frame ${idx}" class="debug-frame-img" />
                    ${frame.face_detected ? `<div class="face-bbox-overlay" style="${faceBoxStyle}"></div>` : ''}
                </div>
                <div class="debug-frame-info">
                    <span class="frame-index">Frame ${idx + 1}</span>
                    ${faceIndicator}
                </div>
            </div>
        `;
    }).join('');
}

function displayDebugTranscript(transcript, length) {
    const container = document.getElementById('debugTranscript');
    if (!container) return;

    const textEl = container.querySelector('.transcript-text');
    const lengthEl = container.querySelector('.transcript-length');

    if (textEl) {
        textEl.textContent = transcript || '(No transcript available)';
    }

    if (lengthEl) {
        lengthEl.textContent = length ? `${length} characters` : '';
    }
}

function displayDebugWaveform(waveformData) {
    const container = document.getElementById('debugWaveform');
    if (!container) return;

    const canvas = document.getElementById('waveformCanvas');
    const durationEl = container.querySelector('.waveform-duration');

    if (!waveformData || !canvas) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'block';

    // Set duration
    if (durationEl) {
        durationEl.textContent = `Duration: ${waveformData.duration_seconds}s`;
    }

    // Draw waveform on canvas
    const ctx = canvas.getContext('2d');
    const width = canvas.parentElement.clientWidth || 600;
    const height = 100;

    canvas.width = width;
    canvas.height = height;

    const points = waveformData.waveform_points || [];
    const rms = waveformData.rms_envelope || [];

    if (points.length === 0) return;

    // Clear canvas
    ctx.fillStyle = '#f9fafb';
    ctx.fillRect(0, 0, width, height);

    // Draw waveform
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 1;
    ctx.beginPath();

    const step = width / points.length;
    const midY = height / 2;

    for (let i = 0; i < points.length; i++) {
        const x = i * step;
        const y = midY - (points[i] * midY * 0.8);

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();

    // Draw RMS envelope (smoother visualization)
    if (rms.length > 0) {
        ctx.strokeStyle = 'rgba(249, 115, 22, 0.5)';
        ctx.lineWidth = 2;
        ctx.beginPath();

        const rmsStep = width / rms.length;

        for (let i = 0; i < rms.length; i++) {
            const x = i * rmsStep;
            const y = height - (rms[i] * height * 0.8);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
    }
}

function downloadResults() {
    if (!currentResults) {
        alert('No results to download');
        return;
    }

    // Create a formatted text report
    let report = '====================================\n';
    report += 'OCEAN PERSONALITY ANALYSIS REPORT\n';
    report += '====================================\n\n';

    const insights = currentResults.insights || {};
    const interpretations = currentResults.interpretations || {};
    const summary = currentResults.summary || {};

    // Add personality title and description
    if (insights.title) {
        report += `PERSONALITY TYPE: ${insights.title}\n`;
        report += '------------------------------------\n\n';

        if (insights.tags && insights.tags.length > 0) {
            report += 'Key Traits: ';
            report += insights.tags.map(t => `${t.emoji} ${t.label}`).join(' | ');
            report += '\n\n';
        }

        if (insights.description) {
            report += insights.description + '\n\n';
        }

        report += '====================================\n\n';
    }

    report += 'DETAILED SCORES\n';
    report += '---------------\n\n';

    const traitOrder = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'];
    traitOrder.forEach(trait => {
        const config = TRAIT_CONFIG[trait];
        const interp = interpretations[trait] || {};
        const percentage = Math.round(interp.t_score || 50);
        report += `${config.emoji} ${config.label}: ${percentage}%\n`;
    });

    report += '\n====================================\n';
    report += 'DISCLAIMER\n';
    report += '====================================\n';
    report += 'These predictions are based on facial appearance alone\n';
    report += 'and should be interpreted with caution. Personality is\n';
    report += 'complex and cannot be accurately determined from photos alone.\n';

    // Download as text file
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `personality-report-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ============================================
// Video Recording Feature
// ============================================

let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let recordingTimer = null;
let countdownTimer = null;
let recordingStartTime = null;
let currentQuestionTimeLimit = 45; // Dynamic per question

// Toggle video requirements accordion items
function toggleRequirement(itemId) {
    const item = document.getElementById(itemId);
    if (!item) return;

    const content = item.querySelector('.requirement-content');
    const isExpanded = content.classList.contains('expanded');

    // Close all other items
    document.querySelectorAll('.requirement-item .requirement-content').forEach(c => {
        c.classList.remove('expanded');
    });
    document.querySelectorAll('.requirement-item').forEach(i => {
        i.classList.remove('expanded');
    });

    // Toggle this item
    if (!isExpanded) {
        content.classList.add('expanded');
        item.classList.add('expanded');
    }
}

// Switch between upload and record modes
function switchInputMode(mode) {
    const uploadCard = document.getElementById('uploadCard');
    const videoRequirements = document.querySelector('.video-requirements');
    const recordArea = document.getElementById('recordArea');
    const previewSection = document.getElementById('previewSection');
    const uploadPageTitle = document.querySelector('.upload-page-title');
    const uploadSection = document.getElementById('uploadSection');

    // Hide preview when switching modes
    if (previewSection) previewSection.style.display = 'none';

    if (mode === 'upload') {
        if (uploadCard) uploadCard.style.display = 'block';
        if (videoRequirements) videoRequirements.style.display = 'block';
        if (recordArea) recordArea.style.display = 'none';
        if (uploadPageTitle) uploadPageTitle.style.display = '';
        if (uploadSection) uploadSection.classList.remove('record-mode');
        document.body.classList.remove('record-view');
        stopCamera();
        resetAssessment();
    } else if (mode === 'record') {
        if (uploadCard) uploadCard.style.display = 'none';
        if (videoRequirements) videoRequirements.style.display = 'none';
        if (recordArea) recordArea.style.display = 'block';
        if (uploadPageTitle) uploadPageTitle.style.display = 'none';
        if (uploadSection) uploadSection.classList.add('record-mode');
        document.body.classList.add('record-view');
        // Show welcome screen, don't start camera yet
        showAssessmentWelcome();
    }
}

// ============================================
// Assessment Flow Functions (Single Compound Question)
// ============================================

// Show the assessment welcome screen
function showAssessmentWelcome() {
    const welcomeEl = document.getElementById('assessmentWelcome');
    const containerEl = document.getElementById('assessmentContainer');
    const summaryEl = document.getElementById('assessmentSummary');

    if (welcomeEl) welcomeEl.style.display = 'block';
    if (containerEl) containerEl.style.display = 'none';
    if (summaryEl) summaryEl.style.display = 'none';
    stopCamera();
}

// Start the assessment
async function startAssessment() {
    // Reset assessment state for 5-question flow
    assessmentState = {
        isActive: true,
        currentQuestionIndex: 0,
        totalQuestions: ASSESSMENT_QUESTIONS.length,
        questionStartTimes: [],
        recordedBlob: null,
        totalDuration: 0,
        questionsAnswered: 0,
        questionsSkipped: 0
    };

    // Hide welcome, show assessment container
    document.getElementById('assessmentWelcome').style.display = 'none';
    document.getElementById('assessmentContainer').style.display = 'flex';

    // Initialize camera
    await startCamera();

    // Display the first question
    displayCurrentQuestion();
}

// Display the current question (overlay on camera)
function displayCurrentQuestion() {
    const question = ASSESSMENT_QUESTIONS[assessmentState.currentQuestionIndex];
    if (!question) return;

    // Update progress indicator
    document.getElementById('questionProgress').textContent =
        `${question.number}/${assessmentState.totalQuestions}`;

    // Update question text
    document.getElementById('questionText').textContent = question.question;

    // Update hint text (if available)
    const hintEl = document.getElementById('questionHint');
    if (hintEl) {
        hintEl.textContent = question.hint || '';
        hintEl.style.display = question.hint ? 'block' : 'none';
    }

    // Clear any inline background (use CSS styles instead)
    const questionCard = document.getElementById('questionCard');
    if (questionCard) {
        questionCard.style.background = '';
    }

    // Set time limit for this question
    currentQuestionTimeLimit = question.timeLimit;

    // Update timer display
    document.getElementById('timerText').textContent = formatTime(question.timeLimit);

    // Show appropriate controls
    if (isRecording) {
        showControlsState('recording');
    } else {
        showControlsState('before');
    }

    // Show camera preview
    document.getElementById('cameraPreview').style.display = 'block';
    document.getElementById('answerPreview').style.display = 'none';
}

// Show the appropriate control state
function showControlsState(state) {
    document.getElementById('controlsBefore').style.display = state === 'before' ? 'flex' : 'none';
    document.getElementById('controlsReading').style.display = state === 'reading' ? 'flex' : 'none';
    document.getElementById('controlsRecording').style.display = state === 'recording' ? 'flex' : 'none';
    document.getElementById('controlsAfter').style.display = state === 'after' ? 'flex' : 'none';
}

// Start recording with countdown (first question only)
function startQuestionRecording() {
    const countdownOverlay = document.getElementById('countdownOverlay');
    const countdownNumber = document.getElementById('countdownNumber');

    countdownOverlay.style.display = 'flex';

    let count = 3;
    countdownNumber.textContent = count;

    countdownTimer = setInterval(() => {
        count--;
        if (count > 0) {
            countdownNumber.textContent = count;
        } else {
            clearInterval(countdownTimer);
            countdownOverlay.style.display = 'none';
            beginContinuousRecording();
        }
    }, 1000);
}

// Begin continuous recording (records through all questions)
function beginContinuousRecording() {
    if (!mediaStream) {
        alert('Camera not available. Please allow camera access.');
        return;
    }

    recordedChunks = [];

    // Create MediaRecorder
    const options = { mimeType: 'video/webm;codecs=vp9,opus' };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options.mimeType = 'video/webm;codecs=vp8,opus';
    }
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options.mimeType = 'video/webm';
    }

    try {
        mediaRecorder = new MediaRecorder(mediaStream, options);
    } catch (e) {
        console.error('MediaRecorder error:', e);
        alert('Recording not supported in this browser.');
        return;
    }

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        finishAssessmentRecording();
    };

    mediaRecorder.start(100);
    isRecording = true;
    recordingStartTime = Date.now();

    // Mark start time for first question
    assessmentState.questionStartTimes[0] = 0;

    // Show recording UI
    showControlsState('recording');
    document.getElementById('recordingTimer').style.display = 'flex';

    // Start timer display for current question
    questionTimerStart = Date.now();
    updateQuestionTimerDisplay();
    recordingTimer = setInterval(updateQuestionTimerDisplay, 100);

    // Start real-time OCEAN scoring (with audio if available)
    const cameraPreview = document.getElementById('cameraPreview');
    if (cameraPreview && !realtimeScoring) {
        showRealtimePanel();
        realtimeScoring = new RealtimeScoring();
        realtimeScoring.start(cameraPreview, mediaStream);  // Pass mediaStream for audio
    }
}

// Timer for current question (counts down from question time limit)
let questionTimerStart = null;

function updateQuestionTimerDisplay() {
    if (!questionTimerStart) return;

    const question = ASSESSMENT_QUESTIONS[assessmentState.currentQuestionIndex];
    if (!question) return;

    const elapsed = (Date.now() - questionTimerStart) / 1000;
    const remaining = Math.max(0, question.timeLimit - elapsed);

    document.getElementById('timerText').textContent = formatTime(Math.ceil(remaining));

    // Auto-advance when time runs out
    if (remaining <= 0 && isRecording) {
        goToNextQuestion();
    }
}

// Go to next question (pause for user to read before continuing)
function goToNextQuestion() {
    const currentIndex = assessmentState.currentQuestionIndex;
    const totalElapsed = (Date.now() - recordingStartTime) / 1000;

    // Mark this question as answered
    assessmentState.questionsAnswered++;

    // Check if there are more questions
    if (currentIndex < assessmentState.totalQuestions - 1) {
        // Move to next question
        assessmentState.currentQuestionIndex++;

        // Record start time for next question (will be updated when user clicks Continue)
        assessmentState.questionStartTimes[assessmentState.currentQuestionIndex] = totalElapsed;

        // PAUSE the timer - don't start counting down yet
        clearInterval(recordingTimer);
        recordingTimer = null;
        questionTimerStart = null;

        // Hide timer while user reads
        document.getElementById('recordingTimer').style.display = 'none';

        // Display the next question (overlay updates)
        displayCurrentQuestion();

        // Show "reading" controls - user must click Continue to start timer
        showControlsState('reading');
    } else {
        // Last question - stop recording and show finish
        stopContinuousRecording();
    }
}

// Continue recording after user has read the question
function continueRecording() {
    // Update the start time for this question segment
    const totalElapsed = (Date.now() - recordingStartTime) / 1000;
    assessmentState.questionStartTimes[assessmentState.currentQuestionIndex] = totalElapsed;

    // Start the timer for this question
    questionTimerStart = Date.now();

    // Show timer
    document.getElementById('recordingTimer').style.display = 'flex';

    // Start timer updates
    updateQuestionTimerDisplay();
    recordingTimer = setInterval(updateQuestionTimerDisplay, 100);

    // Switch to recording controls
    showControlsState('recording');
}

// Stop continuous recording (at the end of all questions)
function stopContinuousRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;

        clearInterval(recordingTimer);
        recordingTimer = null;
        questionTimerStart = null;

        document.getElementById('recordingTimer').style.display = 'none';

        // Stop real-time OCEAN scoring
        if (realtimeScoring) {
            realtimeScoring.stop();
            realtimeScoring = null;
        }
        hideRealtimePanel();
    }
}

// Finish assessment recording - called when MediaRecorder stops
function finishAssessmentRecording() {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const totalDuration = (Date.now() - recordingStartTime) / 1000;

    // Store the recording
    assessmentState.recordedBlob = blob;
    assessmentState.totalDuration = totalDuration;

    // Capture thumbnail from camera before hiding it
    const cameraPreview = document.getElementById('cameraPreview');
    if (cameraPreview && cameraPreview.videoWidth > 0) {
        try {
            const canvas = document.createElement('canvas');
            canvas.width = cameraPreview.videoWidth;
            canvas.height = cameraPreview.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(cameraPreview, 0, 0);
            videoThumbnail = canvas.toDataURL('image/jpeg', 0.8);
        } catch (e) {
            console.warn('Could not capture camera thumbnail:', e);
        }
    }

    // Show preview of recorded answer
    const answerPreview = document.getElementById('answerPreview');
    answerPreview.src = URL.createObjectURL(blob);
    answerPreview.style.display = 'block';
    document.getElementById('cameraPreview').style.display = 'none';

    // Hide question overlay
    document.getElementById('questionOverlay').style.display = 'none';

    // Show "after recording" controls (Analyze / Start Over)
    showControlsState('after');
}

// Cancel assessment
function cancelAssessment() {
    if (isRecording) {
        if (!confirm('Are you sure you want to cancel? Your recording will be lost.')) {
            return;
        }
        // Stop recording without saving
        if (mediaRecorder) {
            mediaRecorder.stop();
        }
        isRecording = false;
        clearInterval(recordingTimer);
        recordingTimer = null;
        if (realtimeScoring) {
            realtimeScoring.stop();
            realtimeScoring = null;
        }
    }
    // Go back to welcome screen
    restartAssessment();
}

// Skip current question (advance without waiting for timer)
function skipQuestion() {
    assessmentState.questionsSkipped++;
    goToNextQuestion();
}

// Re-record the entire assessment
function rerecordAnswer() {
    restartAssessment();
}

// Show assessment summary before submission (not used in new flow - controls show inline)
function showAssessmentSummary() {
    // In the new flow, summary controls are shown inline after recording finishes
    // This function is kept for compatibility
}

// Restart the assessment from the beginning
function restartAssessment() {
    resetAssessment();
    showAssessmentWelcome();
}

// Reset assessment state
function resetAssessment() {
    assessmentState = {
        isActive: false,
        currentQuestionIndex: 0,
        totalQuestions: ASSESSMENT_QUESTIONS.length,
        questionStartTimes: [],
        recordedBlob: null,
        totalDuration: 0,
        questionsAnswered: 0,
        questionsSkipped: 0
    };

    // Clear any recordings
    if (isRecording) {
        stopContinuousRecording();
    }

    // Reset UI elements
    const questionOverlay = document.getElementById('questionOverlay');
    if (questionOverlay) questionOverlay.style.display = 'flex';

    // Reset preview
    const answerPreview = document.getElementById('answerPreview');
    if (answerPreview) {
        answerPreview.style.display = 'none';
        answerPreview.src = '';
    }

    const cameraPreview = document.getElementById('cameraPreview');
    if (cameraPreview) cameraPreview.style.display = 'block';
}

// Update timer display during assessment recording
function updateAssessmentTimerDisplay() {
    const timerText = document.getElementById('timerText');
    if (!timerText || !recordingStartTime) return;

    const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
    const remaining = Math.max(0, currentQuestionTimeLimit - elapsed);

    timerText.textContent = formatTime(remaining);

    // Change color when time is running low
    const timerEl = document.getElementById('recordingTimer');
    if (remaining <= 15) {
        timerEl.classList.add('timer-warning');
    } else {
        timerEl.classList.remove('timer-warning');
    }
}

// Format seconds to MM:SS
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// Submit the assessment for analysis
async function submitAssessment() {
    if (!assessmentState.recordedBlob) {
        alert('Please record your answers before submitting.');
        return;
    }

    // Build the Superpower Question compound format
    // This enables the backend's specialized verbal cues analysis
    const compoundQuestion = {
        full_question: "If you had a superpower, what would it be, how would you use it in your daily life, and what would you do when it doesn't work?",
        parts: ASSESSMENT_QUESTIONS.map((q, index) => {
            const startTime = assessmentState.questionStartTimes[index] || 0;
            const nextStartTime = assessmentState.questionStartTimes[index + 1] || assessmentState.totalDuration;

            return {
                prompt: q.question,
                trait: q.trait,
                signals: [q.hint],
                start_time: startTime,
                end_time: nextStartTime
            };
        })
    };

    const questionMetadata = {
        compound_question: compoundQuestion,
        total_parts: ASSESSMENT_QUESTIONS.length,
        questions_answered: assessmentState.questionsAnswered
    };

    // Show loading and reset progress - hide everything else
    document.getElementById('assessmentContainer').style.display = 'none';
    document.getElementById('recordArea').style.display = 'none';
    document.getElementById('uploadSection').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'flex';

    // Scroll to top so loading is visible
    window.scrollTo({ top: 0, behavior: 'instant' });

    resetProgress();

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', assessmentState.recordedBlob, 'assessment.webm');
        formData.append('question_responses', JSON.stringify(questionMetadata));

        // Include real-time scores if available (for more accurate final analysis)
        if (realtimeScoring) {
            const realtimeData = realtimeScoring.getFullScoreHistory();
            if (realtimeData.sampleCount >= 5) {
                // Only send if we have at least 5 samples
                formData.append('realtime_scores', JSON.stringify(realtimeData));
                console.log(`Sending ${realtimeData.sampleCount} real-time score samples for final analysis`);
            }
        }

        // Use SSE streaming endpoint for real-time progress
        const results = await analyzeWithProgress(formData, {
            use_tta: true,
            method: 'face+middle',
            generate_report: true
        });

        currentResults = results;

        // Hide loading, show results
        document.getElementById('loadingSection').style.display = 'none';
        displayResults(results);

    } catch (error) {
        console.error('Error:', error);
        document.getElementById('loadingSection').style.display = 'none';
        document.getElementById('assessmentSummary').style.display = 'block';
        alert(`Analysis failed: ${error.message}`);
    }
}

// Start the camera for preview
async function startCamera() {
    const cameraPreview = document.getElementById('cameraPreview');

    try {
        // Request camera and microphone access
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            },
            audio: true
        });

        cameraPreview.srcObject = mediaStream;

    } catch (error) {
        console.error('Camera access error:', error);
        alert('Camera access denied. Please allow camera access to use the assessment.');
    }
}

// Stop the camera
function stopCamera() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    const cameraPreview = document.getElementById('cameraPreview');
    if (cameraPreview) {
        cameraPreview.srcObject = null;
    }
}

// Toggle recording on/off
function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startCountdown();
    }
}

// Start countdown before recording
function startCountdown() {
    const countdownOverlay = document.getElementById('countdownOverlay');
    const countdownNumber = document.getElementById('countdownNumber');
    const recordBtn = document.getElementById('recordBtn');

    recordBtn.disabled = true;
    countdownOverlay.style.display = 'flex';

    let count = 3;
    countdownNumber.textContent = count;

    countdownTimer = setInterval(() => {
        count--;
        if (count > 0) {
            countdownNumber.textContent = count;
        } else {
            clearInterval(countdownTimer);
            countdownOverlay.style.display = 'none';
            startRecording();
        }
    }, 1000);
}

// Start recording
function startRecording() {
    if (!mediaStream) {
        alert('Camera not available. Please allow camera access.');
        return;
    }

    recordedChunks = [];

    // Create MediaRecorder with appropriate settings
    const options = { mimeType: 'video/webm;codecs=vp9,opus' };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options.mimeType = 'video/webm;codecs=vp8,opus';
    }
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options.mimeType = 'video/webm';
    }

    try {
        mediaRecorder = new MediaRecorder(mediaStream, options);
    } catch (e) {
        console.error('MediaRecorder error:', e);
        alert('Recording not supported in this browser.');
        return;
    }

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        processRecording();
    };

    mediaRecorder.start(100); // Collect data every 100ms
    isRecording = true;
    recordingStartTime = Date.now();

    // Update UI
    updateRecordingUI(true);

    // Start timer display
    updateTimerDisplay();
    recordingTimer = setInterval(updateTimerDisplay, 100);

    // Auto-stop after RECORDING_DURATION seconds
    setTimeout(() => {
        if (isRecording) {
            stopRecording();
        }
    }, RECORDING_DURATION * 1000);
}

// Stop recording
function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;

        clearInterval(recordingTimer);
        recordingTimer = null;

        updateRecordingUI(false);
    }
}

// Update UI during recording
function updateRecordingUI(recording) {
    const recordBtn = document.getElementById('recordBtn');
    const recordText = recordBtn.querySelector('.record-text');
    const recordIcon = recordBtn.querySelector('.record-icon');
    const recordingTimerEl = document.getElementById('recordingTimer');
    const recordHint = document.getElementById('recordHint');

    recordBtn.disabled = false;

    if (recording) {
        recordBtn.classList.add('recording');
        recordText.textContent = 'Stop Recording';
        recordIcon.classList.add('recording');
        recordingTimerEl.style.display = 'flex';
        recordHint.textContent = 'Recording in progress...';
    } else {
        recordBtn.classList.remove('recording');
        recordText.textContent = 'Start Recording';
        recordIcon.classList.remove('recording');
        recordingTimerEl.style.display = 'none';
        recordHint.textContent = 'Record a 20-second video for personality analysis';
    }
}

// Update the timer display
function updateTimerDisplay() {
    const timerText = document.getElementById('timerText');
    if (!timerText || !recordingStartTime) return;

    const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
    const remaining = Math.max(0, RECORDING_DURATION - elapsed);
    const minutes = Math.floor(remaining / 60);
    const seconds = remaining % 60;

    timerText.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

// Process the recorded video
function processRecording() {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });

    // Create a File object from the Blob
    const file = new File([blob], `recording-${Date.now()}.webm`, { type: 'video/webm' });

    // Set as selected file and show preview
    selectedFile = file;

    // Stop camera
    stopCamera();

    // Show preview
    showRecordedPreview(blob);
}

// Show preview of recorded video
function showRecordedPreview(blob) {
    const previewSection = document.getElementById('previewSection');
    const imagePreview = document.getElementById('imagePreview');
    const videoPreview = document.getElementById('videoPreview');
    const recordArea = document.getElementById('recordArea');

    // Hide record area
    recordArea.style.display = 'none';

    // Show preview section
    previewSection.style.display = 'block';

    // Create URL for the blob
    const videoUrl = URL.createObjectURL(blob);

    // Show video preview
    videoPreview.src = videoUrl;
    videoPreview.style.display = 'block';
    imagePreview.style.display = 'none';

    // Reset video thumbnail
    videoThumbnail = null;

    // Capture thumbnail (one-time handler)
    videoPreview.onloadeddata = null; // Clear any previous handler
    videoPreview.addEventListener('loadeddata', function onRecordedLoaded() {
        videoPreview.removeEventListener('loadeddata', onRecordedLoaded);
        captureVideoThumbnail(videoPreview);
    });
}

// Override resetUpload to also reset recording state
const originalResetUpload = resetUpload;
resetUpload = function() {
    // Stop any recording
    if (isRecording) {
        stopRecording();
    }
    stopCamera();

    // Reset recording state
    recordedChunks = [];
    isRecording = false;

    // Reset to upload mode
    const uploadTab = document.getElementById('uploadTab');
    const recordTab = document.getElementById('recordTab');
    const uploadArea = document.getElementById('uploadArea');
    const recordArea = document.getElementById('recordArea');

    if (uploadTab && recordTab) {
        uploadTab.classList.add('active');
        recordTab.classList.remove('active');
    }

    if (uploadArea) uploadArea.style.display = 'block';
    if (recordArea) recordArea.style.display = 'none';

    // Call original reset
    selectedFile = null;
    currentResults = null;
    videoThumbnail = null;

    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('fileInput').value = '';

    document.getElementById('imagePreview').removeAttribute('src');
    document.getElementById('videoPreview').removeAttribute('src');

    document.getElementById('insightsPhoto').removeAttribute('src');
    const insightsVid = document.getElementById('insightsVideo');
    insightsVid.removeAttribute('src');
    insightsVid.load();

    if (radarChart) {
        radarChart.destroy();
        radarChart = null;
    }
};

// ============================================
// Real-Time OCEAN Scoring Feature
// ============================================

const REALTIME_FPS = 3; // 3 frames per second (one every ~333ms)
const MAX_HISTORY_POINTS = 25; // Keep last 25 data points for chart display
const MAX_FULL_HISTORY = 300; // Store up to 300 frames (5 minutes) for final analysis
const CAPTURE_WIDTH = 640;
const CAPTURE_HEIGHT = 480;
const JPEG_QUALITY = 0.7;

// Trait colors for the time-series chart
const REALTIME_TRAIT_COLORS = {
    openness: '#f59e0b',       // Amber
    conscientiousness: '#3b82f6', // Blue
    extraversion: '#ef4444',   // Red
    agreeableness: '#22c55e',  // Green
    neuroticism: '#8b5cf6'     // Purple
};

// Audio capture settings
const AUDIO_SAMPLE_RATE = 16000;  // 16kHz is sufficient for voice
const AUDIO_BUFFER_DURATION = 2;  // 2 seconds of audio context
const AUDIO_CHUNK_DURATION = 1;   // Send 1 second chunks (matched to frame rate)

// Global real-time scoring instance
let realtimeScoring = null;
let realtimeChart = null;

/**
 * Lightweight audio feature extraction using only AnalyserNode
 * No raw sample buffering - uses Web Audio API's built-in analysis
 * Much more efficient than ScriptProcessorNode approach
 */
class RealtimeAudioCapture {
    constructor(mediaStream) {
        this.mediaStream = mediaStream;
        this.audioContext = null;
        this.analyser = null;
        this.sourceNode = null;
        this.isActive = false;
        this.lastFeatures = null;

        // Pre-allocate typed arrays for efficiency
        this.frequencyData = null;
        this.timeData = null;

        // Running RMS for smoother energy estimation
        this.rmsHistory = [];
        this.rmsHistoryMax = 10; // Keep last 10 readings (~1 second at 10fps)
    }

    /**
     * Start audio capture and analysis
     */
    async start() {
        try {
            // Create audio context (use default sample rate for efficiency)
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            // Create source from media stream
            this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

            // Create analyser for frequency/time domain analysis
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 1024;  // Smaller FFT = faster, 512 bins
            this.analyser.smoothingTimeConstant = 0.5;  // More smoothing = less jitter

            // Pre-allocate typed arrays
            this.frequencyData = new Uint8Array(this.analyser.frequencyBinCount);
            this.timeData = new Uint8Array(this.analyser.fftSize);

            // Connect: source -> analyser (no output needed, just analysis)
            this.sourceNode.connect(this.analyser);
            // Don't connect to destination - avoids feedback

            this.isActive = true;
            console.log('RealtimeAudioCapture started (lightweight mode)');

        } catch (e) {
            console.error('Failed to start audio capture:', e);
        }
    }

    /**
     * Stop audio capture
     */
    stop() {
        this.isActive = false;

        if (this.analyser) {
            this.analyser.disconnect();
        }
        if (this.sourceNode) {
            this.sourceNode.disconnect();
        }
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
        }

        this.rmsHistory = [];
        this.lastFeatures = null;
        console.log('RealtimeAudioCapture stopped');
    }

    /**
     * Extract audio features using only AnalyserNode data
     * Fast and efficient - no sample buffer needed
     */
    extractFeatures() {
        if (!this.isActive || !this.analyser) {
            return null;
        }

        try {
            // Get frequency domain data
            this.analyser.getByteFrequencyData(this.frequencyData);

            // Get time domain data for RMS and ZCR
            this.analyser.getByteTimeDomainData(this.timeData);

            const sampleRate = this.audioContext.sampleRate;
            const binCount = this.analyser.frequencyBinCount;
            const binFrequency = (sampleRate / 2) / binCount;

            // 1. RMS Energy from time domain data (0-255 range, 128 = zero)
            let sumSquares = 0;
            for (let i = 0; i < this.timeData.length; i++) {
                const sample = (this.timeData[i] - 128) / 128;  // Normalize to -1 to 1
                sumSquares += sample * sample;
            }
            const rmsEnergy = Math.sqrt(sumSquares / this.timeData.length);

            // Track RMS history for variance calculation
            this.rmsHistory.push(rmsEnergy);
            if (this.rmsHistory.length > this.rmsHistoryMax) {
                this.rmsHistory.shift();
            }

            // 2. Energy variance from RMS history
            let energyVariance = 0;
            if (this.rmsHistory.length >= 3) {
                const meanRms = this.rmsHistory.reduce((a, b) => a + b, 0) / this.rmsHistory.length;
                let varianceSum = 0;
                for (const rms of this.rmsHistory) {
                    varianceSum += (rms - meanRms) * (rms - meanRms);
                }
                energyVariance = Math.sqrt(varianceSum / this.rmsHistory.length);
            }

            // 3. Zero Crossing Rate from time domain
            let zeroCrossings = 0;
            for (let i = 1; i < this.timeData.length; i++) {
                if ((this.timeData[i] >= 128 && this.timeData[i-1] < 128) ||
                    (this.timeData[i] < 128 && this.timeData[i-1] >= 128)) {
                    zeroCrossings++;
                }
            }
            const zcr = zeroCrossings / this.timeData.length;

            // 4. Spectral centroid and spread from frequency data
            let weightedSum = 0;
            let totalMagnitude = 0;

            for (let i = 0; i < binCount; i++) {
                const magnitude = this.frequencyData[i];
                const frequency = i * binFrequency;
                weightedSum += frequency * magnitude;
                totalMagnitude += magnitude;
            }

            let spectralCentroid = 0;
            let spectralSpread = 0;

            if (totalMagnitude > 100) {  // Minimum energy threshold
                spectralCentroid = weightedSum / totalMagnitude;

                // Spectral spread
                let spreadSum = 0;
                for (let i = 0; i < binCount; i++) {
                    const frequency = i * binFrequency;
                    const diff = frequency - spectralCentroid;
                    spreadSum += diff * diff * this.frequencyData[i];
                }
                spectralSpread = Math.sqrt(spreadSum / totalMagnitude);
            }

            // 5. Voice Activity Detection
            // Lower threshold for better sensitivity - web audio RMS is typically low
            const voiceThreshold = 0.005;  // Lowered from 0.02

            // Voice is active if we have some energy and spectral content
            // Spectral centroid check is relaxed - just needs to be non-zero
            const isVoiceActive = rmsEnergy > voiceThreshold && totalMagnitude > 50;

            // Debug logging (every call to help diagnose)
            if (Math.random() < 0.1) {  // Log 10% of the time to avoid spam
                console.log(`[Audio] RMS: ${rmsEnergy.toFixed(4)}, ` +
                           `Centroid: ${spectralCentroid.toFixed(0)}Hz, ` +
                           `TotalMag: ${totalMagnitude.toFixed(0)}, ` +
                           `Voice: ${isVoiceActive}`);
            }

            this.lastFeatures = {
                rms_energy: rmsEnergy,
                energy_variance: energyVariance,
                zcr: zcr,
                voice_active: isVoiceActive,
                spectral_centroid: spectralCentroid,
                spectral_spread: spectralSpread,
                sample_count: this.timeData.length
            };

            return this.lastFeatures;

        } catch (e) {
            console.warn('Audio feature extraction error:', e);
            return null;
        }
    }

    /**
     * Get last extracted features (for sending with frames)
     */
    getLastFeatures() {
        return this.lastFeatures;
    }
}

/**
 * Captures frames from a video element at regular intervals
 */
class RealtimeFrameCapture {
    constructor(videoElement) {
        this.video = videoElement;
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.frameIndex = 0;
        this.intervalId = null;
        this.onFrame = null;

        // Set canvas size
        this.canvas.width = CAPTURE_WIDTH;
        this.canvas.height = CAPTURE_HEIGHT;
    }

    /**
     * Start capturing frames
     * @param {Function} onFrameCallback - Called with each frame
     */
    start(onFrameCallback) {
        this.onFrame = onFrameCallback;
        this.frameIndex = 0;

        // Capture first frame immediately
        this.captureFrame();

        // Then capture every second
        this.intervalId = setInterval(() => {
            this.captureFrame();
        }, 1000 / REALTIME_FPS);
    }

    /**
     * Stop capturing frames
     */
    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }

    /**
     * Capture a single frame and send to callback
     */
    captureFrame() {
        if (!this.video || this.video.readyState < 2) return;

        try {
            // Draw video frame to canvas (scaled)
            this.ctx.drawImage(
                this.video,
                0, 0,
                this.canvas.width,
                this.canvas.height
            );

            // Get JPEG base64
            const dataUrl = this.canvas.toDataURL('image/jpeg', JPEG_QUALITY);
            const base64Data = dataUrl.split(',')[1];

            if (this.onFrame && base64Data) {
                this.onFrame({
                    data: base64Data,
                    frameIndex: this.frameIndex++,
                    timestamp: Date.now()
                });
            }
        } catch (e) {
            console.warn('Frame capture error:', e);
        }
    }
}

/**
 * Manages WebSocket connection and real-time scoring
 */
class RealtimeScoring {
    constructor() {
        this.socket = null;
        this.frameCapture = null;
        this.audioCapture = null;  // NEW: Audio capture instance
        this.mediaStream = null;   // NEW: Store media stream for audio
        this.isActive = false;
        this.videoElement = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 3;
        this.reconnectDelay = 1000;

        // Debug mode - when enabled, server sends processed face images
        this.debugMode = false;
        this.debugFrames = [];  // Store recent debug frames for display
        this.maxDebugFrames = 10;  // Keep last N frames for visualization
        this.totalFramesProcessed = 0;
        this.framesWithFace = 0;
        this.lastInferenceTime = 0;

        // Smoothing factor for exponential moving average (0.3 = 30% new, 70% previous)
        // Lower = smoother but slower to react, Higher = more responsive but jumpier
        this.smoothingAlpha = 0.3;

        // Smoothed scores for display (reduces jitter)
        this.smoothedScores = {
            openness: null,
            conscientiousness: null,
            extraversion: null,
            agreeableness: null,
            neuroticism: null
        };

        // Score history for chart (capped at MAX_HISTORY_POINTS for display)
        this.scoreHistory = {
            openness: [],
            conscientiousness: [],
            extraversion: [],
            agreeableness: [],
            neuroticism: []
        };
        this.timestamps = [];

        // Full score history for final analysis (capped at MAX_FULL_HISTORY)
        // Stores RAW scores, not smoothed - for accurate final analysis
        this.fullScoreHistory = {
            openness: [],
            conscientiousness: [],
            extraversion: [],
            agreeableness: [],
            neuroticism: []
        };
        this.fullTimestamps = [];

        // Track pending frames to avoid queue buildup
        this.pendingFrame = null;
        this.isProcessing = false;

        // NEW: Audio score history for debugging
        this.audioScoreHistory = {
            openness: [],
            conscientiousness: [],
            extraversion: [],
            agreeableness: [],
            neuroticism: []
        };
        this.videoScoreHistory = {
            openness: [],
            conscientiousness: [],
            extraversion: [],
            agreeableness: [],
            neuroticism: []
        };
    }

    /**
     * Apply exponential moving average smoothing to scores
     * @param {Object} rawScores - Raw scores from model (0-100 scale)
     * @returns {Object} Smoothed scores for display
     */
    applySmoothing(rawScores) {
        const smoothed = {};
        const alpha = this.smoothingAlpha;

        for (const trait in rawScores) {
            const rawScore = rawScores[trait];

            if (this.smoothedScores[trait] === null) {
                // First frame - no previous value to smooth with
                this.smoothedScores[trait] = rawScore;
            } else {
                // Apply exponential moving average: new = α*raw + (1-α)*previous
                this.smoothedScores[trait] = alpha * rawScore + (1 - alpha) * this.smoothedScores[trait];
            }

            smoothed[trait] = this.smoothedScores[trait];
        }

        return smoothed;
    }

    /**
     * Start real-time scoring
     * @param {HTMLVideoElement} videoElement - The video element to capture from
     * @param {MediaStream} mediaStream - Optional media stream for audio capture
     */
    async start(videoElement, mediaStream = null) {
        // Guard: prevent starting if already active
        if (this.isActive) {
            console.log('RealtimeScoring already active, ignoring start');
            return;
        }

        this.videoElement = videoElement;
        this.mediaStream = mediaStream;
        this.resetHistory();

        // Stop any existing audio capture
        if (this.audioCapture) {
            this.audioCapture.stop();
            this.audioCapture = null;
        }

        // Start audio capture if media stream provided
        if (mediaStream && mediaStream.getAudioTracks().length > 0) {
            this.audioCapture = new RealtimeAudioCapture(mediaStream);
            await this.audioCapture.start();
            console.log('Audio capture started for real-time scoring');
        } else {
            console.log('No audio stream available, video-only scoring');
        }

        this.connect();
    }

    /**
     * Connect to WebSocket server
     */
    connect() {
        // Guard: close existing socket before creating a new one
        if (this.socket) {
            console.log('Closing existing WebSocket before reconnecting');
            try {
                this.socket.close();
            } catch (e) {
                // Ignore close errors
            }
            this.socket = null;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/realtime-predict`;

        console.log('Connecting to WebSocket:', wsUrl);
        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = () => {
            console.log('Real-time scoring connected');
            this.reconnectAttempts = 0;
            this.isActive = true;

            // Stop any existing frame capture before starting new one
            if (this.frameCapture) {
                this.frameCapture.stop();
            }

            // Start frame capture
            this.frameCapture = new RealtimeFrameCapture(this.videoElement);
            this.frameCapture.start((frame) => this.sendFrame(frame));

            updateRealtimeStatus('connected', 'Live');
        };

        this.socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleServerMessage(data);
            } catch (e) {
                console.error('Error parsing WebSocket message:', e);
            }
        };

        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateRealtimeStatus('error', 'Connection error');
        };

        this.socket.onclose = () => {
            console.log('WebSocket closed');
            this.isActive = false;

            // Stop frame capture
            if (this.frameCapture) {
                this.frameCapture.stop();
            }

            // Stop audio capture
            if (this.audioCapture) {
                this.audioCapture.stop();
            }

            // Try to reconnect if recording is still active
            if (isRecording && this.reconnectAttempts < this.maxReconnectAttempts) {
                updateRealtimeStatus('reconnecting', 'Reconnecting...');
                setTimeout(() => {
                    this.reconnectAttempts++;
                    console.log(`Reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
                    this.connect();
                }, this.reconnectDelay * this.reconnectAttempts);
            } else if (isRecording) {
                updateRealtimeStatus('disconnected', 'Offline - Analysis after recording');
            }
        };
    }

    /**
     * Stop real-time scoring
     */
    stop() {
        this.isActive = false;

        if (this.frameCapture) {
            this.frameCapture.stop();
            this.frameCapture = null;
        }

        // Stop audio capture
        if (this.audioCapture) {
            this.audioCapture.stop();
            this.audioCapture = null;
        }

        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            try {
                this.socket.send(JSON.stringify({ type: 'stop' }));
            } catch (e) {
                // Ignore send errors during close
            }
            this.socket.close();
        }
        this.socket = null;
    }

    /**
     * Send a frame to the server
     * @param {Object} frame - Frame data with base64 and metadata
     */
    sendFrame(frame) {
        if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;

        // If already processing, store latest frame to avoid queue buildup
        if (this.isProcessing) {
            this.pendingFrame = frame;
            return;
        }

        this.isProcessing = true;

        // Extract audio features if audio capture is active
        let audioFeatures = null;
        if (this.audioCapture) {
            audioFeatures = this.audioCapture.extractFeatures();
            if (audioFeatures && frame.frameIndex % 5 === 0) {
                console.log(`[Frame ${frame.frameIndex}] Audio features:`,
                    `RMS=${audioFeatures.rms_energy?.toFixed(4)}, ` +
                    `Voice=${audioFeatures.voice_active}`);
            }
        } else if (frame.frameIndex === 0) {
            console.log('[Frame 0] No audio capture available');
        }

        try {
            const message = {
                type: 'frame',
                data: frame.data,
                frame_index: frame.frameIndex,
                timestamp_ms: frame.timestamp,
                debug: this.debugMode  // Request debug info (face image, bbox)
            };

            // Include audio features if available
            if (audioFeatures) {
                message.audio_features = audioFeatures;
            }

            this.socket.send(JSON.stringify(message));
        } catch (e) {
            console.error('Error sending frame:', e);
            this.isProcessing = false;
        }
    }

    /**
     * Enable/disable debug mode
     * @param {boolean} enabled - Whether to enable debug mode
     */
    setDebugMode(enabled) {
        this.debugMode = enabled;
        console.log(`Debug mode ${enabled ? 'enabled' : 'disabled'}`);
        if (!enabled) {
            this.debugFrames = [];
        }
    }

    /**
     * Add a debug frame to the visualization
     * @param {Object} frameData - Debug frame data
     */
    addDebugFrame(frameData) {
        this.debugFrames.push(frameData);
        if (this.debugFrames.length > this.maxDebugFrames) {
            this.debugFrames.shift();
        }
    }

    /**
     * Handle message from server
     * @param {Object} data - Parsed JSON message
     */
    handleServerMessage(data) {
        switch (data.type) {
            case 'ready':
                console.log('Server ready:', data.model_info);
                break;

            case 'prediction':
                this.isProcessing = false;

                // Process pending frame if any
                if (this.pendingFrame) {
                    const frame = this.pendingFrame;
                    this.pendingFrame = null;
                    this.sendFrame(frame);
                }

                // Log scores for debugging (now includes video, audio, and fused)
                const vs = data.video_scores || {};
                const as = data.audio_scores || {};
                const fs = data.scores || {};
                console.log(`[FRAME ${data.frame_index}] ` +
                    `Video: O=${vs.openness?.toFixed(1)}, E=${vs.extraversion?.toFixed(1)} | ` +
                    `Audio: O=${as.openness?.toFixed(1) || '-'}, E=${as.extraversion?.toFixed(1) || '-'} | ` +
                    `Fused: O=${fs.openness?.toFixed(1)}, E=${fs.extraversion?.toFixed(1)} | ` +
                    `face=${data.face_detected}, voice=${data.voice_detected}`);

                // Store video and audio scores separately for debug display
                if (data.video_scores) {
                    for (const trait in data.video_scores) {
                        if (this.videoScoreHistory[trait]) {
                            this.videoScoreHistory[trait].push(data.video_scores[trait]);
                            if (this.videoScoreHistory[trait].length > MAX_HISTORY_POINTS) {
                                this.videoScoreHistory[trait].shift();
                            }
                        }
                    }
                }
                if (data.audio_scores) {
                    for (const trait in data.audio_scores) {
                        if (this.audioScoreHistory[trait]) {
                            this.audioScoreHistory[trait].push(data.audio_scores[trait]);
                            if (this.audioScoreHistory[trait].length > MAX_HISTORY_POINTS) {
                                this.audioScoreHistory[trait].shift();
                            }
                        }
                    }
                }

                // Store fused scores in history
                this.updateScoreHistory(data.scores, data.scores, data.timestamp_ms);

                // Update chart with fused scores
                this.updateRealtimeChart();

                // Update UI with fused scores and debug info
                updateCurrentScoresUI(data.scores, data.face_detected, data.voice_detected, data.video_scores, data.audio_scores);

                // Update debug stats
                this.totalFramesProcessed++;
                if (data.face_detected) this.framesWithFace++;
                this.lastInferenceTime = data.inference_time_ms || 0;

                // Handle debug data if present
                if (this.debugMode && data.debug) {
                    this.addDebugFrame({
                        frameIndex: data.frame_index,
                        faceDetected: data.face_detected,
                        faceImage: data.debug.face_image,
                        faceBbox: data.debug.face_bbox,
                        originalSize: data.debug.original_size,
                        processedSize: data.debug.processed_size,
                        scores: data.scores,
                        inferenceTime: data.inference_time_ms
                    });
                }
                break;

            case 'error':
                this.isProcessing = false;
                console.warn('Server error:', data.message);
                updateFaceIndicator(false, data.message);

                // Process pending frame
                if (this.pendingFrame) {
                    const frame = this.pendingFrame;
                    this.pendingFrame = null;
                    this.sendFrame(frame);
                }
                break;

            case 'heartbeat':
            case 'pong':
                // Connection alive
                break;

            default:
                console.log('Unknown message type:', data.type);
        }
    }

    /**
     * Update score history for chart and final analysis
     * @param {Object} rawScores - Raw scores for final analysis
     * @param {Object} smoothedScores - Smoothed scores for chart display
     * @param {number} timestamp - Frame timestamp
     */
    updateScoreHistory(rawScores, smoothedScores, timestamp) {
        for (const trait in rawScores) {
            if (this.scoreHistory[trait]) {
                // Update chart history with SMOOTHED scores (for visual stability)
                this.scoreHistory[trait].push(smoothedScores[trait]);
                if (this.scoreHistory[trait].length > MAX_HISTORY_POINTS) {
                    this.scoreHistory[trait].shift();
                }

                // Update full history with RAW scores (for accurate final analysis)
                this.fullScoreHistory[trait].push(rawScores[trait]);
                if (this.fullScoreHistory[trait].length > MAX_FULL_HISTORY) {
                    this.fullScoreHistory[trait].shift();
                }
            }
        }

        this.timestamps.push(timestamp);
        if (this.timestamps.length > MAX_HISTORY_POINTS) {
            this.timestamps.shift();
        }

        this.fullTimestamps.push(timestamp);
        if (this.fullTimestamps.length > MAX_FULL_HISTORY) {
            this.fullTimestamps.shift();
        }
    }

    /**
     * Update the real-time chart
     */
    updateRealtimeChart() {
        if (realtimeChart) {
            updateRealtimeChartData(this.scoreHistory, this.timestamps);
        }
    }

    /**
     * Get full score history for final analysis
     * @returns {Object} Full score history with all collected data points
     */
    getFullScoreHistory() {
        return {
            scores: this.fullScoreHistory,
            timestamps: this.fullTimestamps,
            sampleCount: this.fullScoreHistory.openness.length
        };
    }

    /**
     * Reset score history
     */
    resetHistory() {
        this.scoreHistory = {
            openness: [],
            conscientiousness: [],
            extraversion: [],
            agreeableness: [],
            neuroticism: []
        };
        this.timestamps = [];

        this.fullScoreHistory = {
            openness: [],
            conscientiousness: [],
            extraversion: [],
            agreeableness: [],
            neuroticism: []
        };
        this.fullTimestamps = [];

        // Reset smoothed scores for fresh start
        this.smoothedScores = {
            openness: null,
            conscientiousness: null,
            extraversion: null,
            agreeableness: null,
            neuroticism: null
        };

        // Reset video/audio score histories for debug display
        this.videoScoreHistory = {
            openness: [],
            conscientiousness: [],
            extraversion: [],
            agreeableness: [],
            neuroticism: []
        };
        this.audioScoreHistory = {
            openness: [],
            conscientiousness: [],
            extraversion: [],
            agreeableness: [],
            neuroticism: []
        };
    }
}

/**
 * Create the real-time Chart.js time-series chart
 */
function createRealtimeChart() {
    const ctx = document.getElementById('realtimeChart');
    if (!ctx) return null;

    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Openness',
                    data: [],
                    borderColor: REALTIME_TRAIT_COLORS.openness,
                    backgroundColor: 'transparent',
                    tension: 0.4,
                    pointRadius: 2,
                    borderWidth: 2
                },
                {
                    label: 'Conscientiousness',
                    data: [],
                    borderColor: REALTIME_TRAIT_COLORS.conscientiousness,
                    backgroundColor: 'transparent',
                    tension: 0.4,
                    pointRadius: 2,
                    borderWidth: 2
                },
                {
                    label: 'Extraversion',
                    data: [],
                    borderColor: REALTIME_TRAIT_COLORS.extraversion,
                    backgroundColor: 'transparent',
                    tension: 0.4,
                    pointRadius: 2,
                    borderWidth: 2
                },
                {
                    label: 'Agreeableness',
                    data: [],
                    borderColor: REALTIME_TRAIT_COLORS.agreeableness,
                    backgroundColor: 'transparent',
                    tension: 0.4,
                    pointRadius: 2,
                    borderWidth: 2
                },
                {
                    label: 'Neuroticism',
                    data: [],
                    borderColor: REALTIME_TRAIT_COLORS.neuroticism,
                    backgroundColor: 'transparent',
                    tension: 0.4,
                    pointRadius: 2,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 300
            },
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 10,
                        font: { size: 10 }
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    min: 0,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Score %',
                        font: { size: 10 }
                    },
                    ticks: {
                        stepSize: 25,
                        font: { size: 10 }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false  // Hidden - color bars above show the traits
                },
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    titleColor: '#374151',
                    bodyColor: '#6b7280',
                    borderColor: '#e5e7eb',
                    borderWidth: 1,
                    padding: 8,
                    bodyFont: { size: 11 },
                    titleFont: { size: 11, weight: '600' },
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(0)}%`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Update the real-time chart with new data
 */
function updateRealtimeChartData(scoreHistory, timestamps) {
    if (!realtimeChart) return;

    const traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'];

    // Update labels (show seconds elapsed)
    realtimeChart.data.labels = timestamps.map((t, i) => `${i + 1}s`);

    // Update datasets
    traits.forEach((trait, index) => {
        realtimeChart.data.datasets[index].data = scoreHistory[trait] || [];
    });

    realtimeChart.update('none');  // No animation for frequent updates
}

/**
 * Update the real-time status indicator
 */
function updateRealtimeStatus(status, text) {
    const indicator = document.getElementById('realtimeStatus');
    if (!indicator) return;

    const dot = indicator.querySelector('.status-dot');
    const textEl = indicator.querySelector('.status-text');

    if (dot) {
        dot.className = `status-dot ${status}`;
    }
    if (textEl) {
        textEl.textContent = text;
    }
}

/**
 * Update the current scores display (supports multiple HTML structures)
 */
function updateCurrentScoresUI(scores, faceDetected, voiceDetected = false, videoScores = null, audioScores = null) {
    const traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'];

    traits.forEach(trait => {
        const score = scores[trait] || 0;
        const percentage = Math.round(score);

        // Try score-pill structure (new bottom panel layout)
        const scorePill = document.querySelector(`.score-pill[data-trait="${trait}"]`);
        if (scorePill) {
            const valueEl = scorePill.querySelector('.score-value');
            if (valueEl) {
                valueEl.textContent = `${percentage}%`;
            }
            return;
        }

        // Try rt-score-item structure (old assessment grid layout)
        const rtScoreItem = document.querySelector(`.rt-score-item[data-trait="${trait}"]`);
        if (rtScoreItem) {
            const fillEl = rtScoreItem.querySelector('.rt-score-fill');
            const valueEl = rtScoreItem.querySelector('.rt-score-value');

            if (fillEl) {
                fillEl.style.width = `${percentage}%`;
            }
            if (valueEl) {
                valueEl.textContent = `${percentage}%`;
            }
            return;
        }

        // Fallback to old ID-based structure
        const fillEl = document.getElementById(`rtScore${capitalizeFirst(trait)}`);
        const valueEl = document.getElementById(`rtValue${capitalizeFirst(trait)}`);

        if (fillEl) {
            fillEl.style.width = `${score}%`;

            // Update color based on score
            if (score >= 70) {
                fillEl.style.backgroundColor = '#22c55e';
            } else if (score >= 40) {
                fillEl.style.backgroundColor = REALTIME_TRAIT_COLORS[trait];
            } else {
                fillEl.style.backgroundColor = '#ef4444';
            }
        }
        if (valueEl) {
            valueEl.textContent = `${Math.round(score)}%`;
        }
    });

    // Update face detection indicator
    updateFaceIndicator(faceDetected);
}


/**
 * Update face detection indicator
 */
function updateFaceIndicator(detected, message = null) {
    // Try new ID first, then fallback to old ID
    const indicator = document.getElementById('faceIndicator') || document.getElementById('faceDetectionIndicator');
    if (!indicator) return;

    const textEl = indicator.querySelector('.face-text') || document.getElementById('faceText');

    if (detected) {
        indicator.classList.remove('error', 'warning');
        if (textEl) textEl.textContent = 'Face detected';
    } else {
        indicator.classList.add('warning');
        if (textEl) textEl.textContent = message || 'No face detected';
    }
}

/**
 * Capitalize first letter
 */
function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Show the real-time panel
 */
function showRealtimePanel() {
    const panel = document.getElementById('realtimePanel');
    if (panel) {
        // Use flex for bottom panel, block for old side panel
        panel.style.display = panel.classList.contains('realtime-bottom-panel') ? 'flex' : 'block';

        // Initialize chart if not already created
        if (!realtimeChart) {
            realtimeChart = createRealtimeChart();
        }

        // Reset status
        updateRealtimeStatus('connecting', 'Connecting...');
    }

}

/**
 * Hide the real-time panel
 */
function hideRealtimePanel() {
    const panel = document.getElementById('realtimePanel');
    if (panel) {
        panel.style.display = 'none';
    }

    // Destroy chart to free memory
    if (realtimeChart) {
        realtimeChart.destroy();
        realtimeChart = null;
    }
}

// ============================================
// Integration with Recording Flow
// ============================================

// Store original startRecording function
const originalStartRecording = startRecording;

// Override startRecording to add real-time scoring
startRecording = function() {
    // Call original function
    originalStartRecording.call(this);

    // Start real-time scoring (with audio if available)
    // Guard against duplicate instances
    const cameraPreview = document.getElementById('cameraPreview');
    if (cameraPreview && !realtimeScoring) {
        showRealtimePanel();
        realtimeScoring = new RealtimeScoring();
        realtimeScoring.start(cameraPreview, mediaStream);  // Pass mediaStream for audio
    }
};

// Store original stopRecording function
const originalStopRecording = stopRecording;

// Override stopRecording to stop real-time scoring
stopRecording = function() {
    // Stop real-time scoring
    if (realtimeScoring) {
        realtimeScoring.stop();
        realtimeScoring = null;
    }
    hideRealtimePanel();

    // Call original function
    originalStopRecording.call(this);
};

// ===== FOOTER FUNCTIONS =====

/**
 * Handle newsletter subscription
 */
function subscribeNewsletter() {
    const input = document.querySelector('.newsletter-input');
    const email = input ? input.value.trim() : '';

    if (!email) {
        return;
    }

    // Show success feedback
    const btn = document.querySelector('.newsletter-btn');
    const originalHTML = btn.innerHTML;

    btn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
            <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
    `;
    btn.style.background = 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)';

    // Clear input
    if (input) {
        input.value = '';
        input.placeholder = 'Thanks for subscribing!';
    }

    // Reset after 3 seconds
    setTimeout(() => {
        btn.innerHTML = originalHTML;
        btn.style.background = '';
        if (input) {
            input.placeholder = 'Enter your email';
        }
    }, 3000);

    console.log('Newsletter subscription:', email);
}

// ============================================
// AI Chat Assistant
// ============================================

/**
 * Initialize the AI Chat Assistant
 */
function initAiAssistant() {
    const floatingBtn = document.getElementById('aiFloatingBtn');
    const chatWindow = document.getElementById('aiChatWindow');
    const chatClose = document.getElementById('aiChatClose');
    const chatInput = document.getElementById('aiChatInput');
    const charCount = document.getElementById('aiCharCount');
    const sendBtn = document.getElementById('aiSendBtn');
    const messagesContainer = document.getElementById('aiChatMessages');

    if (!floatingBtn || !chatWindow) return;

    let isChatOpen = false;
    const maxChars = 2000;

    // Toggle chat window
    function toggleChat() {
        isChatOpen = !isChatOpen;

        if (isChatOpen) {
            chatWindow.classList.add('open');
            floatingBtn.classList.add('active');
            floatingBtn.querySelector('.ai-icon-bot').style.display = 'none';
            floatingBtn.querySelector('.ai-icon-close').style.display = 'block';
            // Focus input when opening
            setTimeout(() => chatInput?.focus(), 300);
        } else {
            chatWindow.classList.remove('open');
            floatingBtn.classList.remove('active');
            floatingBtn.querySelector('.ai-icon-bot').style.display = 'block';
            floatingBtn.querySelector('.ai-icon-close').style.display = 'none';
        }
    }

    // Close chat
    function closeChat() {
        if (isChatOpen) {
            isChatOpen = false;
            chatWindow.classList.remove('open');
            floatingBtn.classList.remove('active');
            floatingBtn.querySelector('.ai-icon-bot').style.display = 'block';
            floatingBtn.querySelector('.ai-icon-close').style.display = 'none';
        }
    }

    // Handle input changes
    function handleInputChange() {
        if (chatInput && charCount) {
            const length = chatInput.value.length;
            charCount.textContent = length;

            // Visual feedback if near limit
            if (length > maxChars * 0.9) {
                charCount.style.color = '#ef4444';
            } else {
                charCount.style.color = '';
            }
        }
    }

    // Message history for multi-turn conversations
    let messageHistory = [];

    // Send message to AI personality coach with streaming
    async function sendMessage() {
        if (!chatInput) return;

        const message = chatInput.value.trim();
        if (!message) return;

        // Check if we have personality data
        if (!currentResults || !currentResults.predictions) {
            addMessageToChat('assistant', 'Please complete a personality analysis first so I can provide personalized coaching advice.');
            return;
        }

        console.log('AI Assistant - Sending message:', message);

        // Add user message to chat
        addMessageToChat('user', message);

        // Clear input
        chatInput.value = '';
        handleInputChange();

        // Create streaming message element
        const streamingEl = document.createElement('div');
        streamingEl.className = 'ai-message assistant streaming';
        streamingEl.textContent = '';
        messagesContainer.appendChild(streamingEl);
        messagesContainer.classList.add('has-messages');
        chatWindow.classList.add('has-messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        let fullResponse = '';

        try {
            // Prepare OCEAN scores from interpretations (T-scores, 0-100 scale)
            // These match what's displayed in the UI
            const interpretations = currentResults.interpretations || {};
            const oceanScores = {
                openness: (interpretations.openness?.t_score || 50) / 100,
                conscientiousness: (interpretations.conscientiousness?.t_score || 50) / 100,
                extraversion: (interpretations.extraversion?.t_score || 50) / 100,
                agreeableness: (interpretations.agreeableness?.t_score || 50) / 100,
                neuroticism: (interpretations.neuroticism?.t_score || 50) / 100
            };

            // Prepare derived metrics if available
            const derivedMetrics = currentResults.derived_metrics || null;

            // Get user's spoken responses from video recording (if available)
            const userTranscript = currentResults.user_transcript || null;

            // Build request payload with full interpretations for personalized insights
            const payload = {
                message: message,
                ocean_scores: oceanScores,
                derived_metrics: derivedMetrics,
                interpretations: interpretations,
                user_transcript: userTranscript,
                message_history: messageHistory.length > 0 ? messageHistory : null
            };

            // Call the streaming chat API
            const response = await fetch(`${API_BASE_URL}/chat/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || 'Failed to get response from coach');
            }

            // Read the stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));

                            if (data.type === 'chunk') {
                                // Append chunk to response
                                fullResponse += data.content;
                                // Parse markdown in real-time during streaming
                                streamingEl.innerHTML = parseMarkdown(fullResponse);
                                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                            } else if (data.type === 'done') {
                                // Streaming complete - final markdown parse
                                streamingEl.classList.remove('streaming');
                                streamingEl.innerHTML = parseMarkdown(fullResponse);
                            } else if (data.type === 'error') {
                                throw new Error(data.message);
                            }
                        } catch (parseError) {
                            // Ignore parse errors for incomplete chunks
                            if (line.slice(6).trim()) {
                                console.warn('SSE parse error:', parseError);
                            }
                        }
                    }
                }
            }

            // Add to message history for context
            messageHistory.push({ role: 'user', content: message });
            messageHistory.push({ role: 'assistant', content: fullResponse });

            // Keep only last 20 messages to avoid context overflow
            if (messageHistory.length > 20) {
                messageHistory = messageHistory.slice(-20);
            }

        } catch (error) {
            console.error('AI Chat error:', error);
            streamingEl.textContent = `Sorry, I encountered an error: ${error.message}. Please try again.`;
            streamingEl.classList.remove('streaming');
            streamingEl.classList.add('error');
        }
    }

    // Simple markdown parser for chat messages
    function parseMarkdown(text) {
        return text
            // Bold: **text** or __text__
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/__(.*?)__/g, '<strong>$1</strong>')
            // Italic: *text* or _text_
            .replace(/\*([^*]+)\*/g, '<em>$1</em>')
            .replace(/_([^_]+)_/g, '<em>$1</em>')
            // Bullet points: - item or • item
            .replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>')
            // Numbered lists: 1. item
            .replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>')
            // Wrap consecutive <li> in <ul>
            .replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
            // Line breaks
            .replace(/\n/g, '<br>');
    }

    // Add message to chat display
    function addMessageToChat(role, content) {
        if (!messagesContainer) return;

        const messageEl = document.createElement('div');
        messageEl.className = `ai-message ${role}`;

        // Parse markdown for assistant messages
        if (role === 'assistant') {
            messageEl.innerHTML = parseMarkdown(content);
        } else {
            messageEl.textContent = content;
        }

        messagesContainer.appendChild(messageEl);
        messagesContainer.classList.add('has-messages');

        // Add has-messages class to chat window for compact styling
        chatWindow.classList.add('has-messages');

        // Smooth scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Handle keyboard shortcuts
    function handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    }

    // Close when clicking outside
    function handleClickOutside(event) {
        if (!chatWindow.contains(event.target) && !floatingBtn.contains(event.target)) {
            closeChat();
        }
    }

    // Event listeners
    floatingBtn.addEventListener('click', toggleChat);
    chatClose?.addEventListener('click', closeChat);
    chatInput?.addEventListener('input', handleInputChange);
    chatInput?.addEventListener('keydown', handleKeyDown);
    sendBtn?.addEventListener('click', sendMessage);
    document.addEventListener('mousedown', handleClickOutside);

    // Escape key to close
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && isChatOpen) {
            closeChat();
        }
    });
}

// Initialize AI Assistant when DOM is ready
document.addEventListener('DOMContentLoaded', initAiAssistant);

/**
 * Open AI Chat with a pre-filled question about a trait
 * Called from "How to improve" buttons in Big 5 accordion
 */
function openAiChatWithQuestion(traitName, event) {
    // Prevent accordion toggle
    if (event) {
        event.stopPropagation();
    }

    const floatingBtn = document.getElementById('aiFloatingBtn');
    const chatWindow = document.getElementById('aiChatWindow');
    const chatInput = document.getElementById('aiChatInput');

    if (!floatingBtn || !chatWindow) return;

    // Open the chat
    chatWindow.classList.add('open');
    floatingBtn.classList.add('active');
    floatingBtn.querySelector('.ai-icon-bot').style.display = 'none';
    floatingBtn.querySelector('.ai-icon-close').style.display = 'block';

    // Pre-fill the input with the question
    if (chatInput) {
        chatInput.value = `How can I improve my ${traitName} score?`;
        chatInput.focus();

        // Update character count
        const charCount = document.getElementById('aiCharCount');
        if (charCount) {
            charCount.textContent = chatInput.value.length;
        }
    }
}
