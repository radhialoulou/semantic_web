// SPARQL Agent LLM - Frontend JavaScript
// Handles form submission, API calls, and dynamic UI updates

const API_BASE_URL = 'http://localhost:5000';

// DOM Elements
const questionForm = document.getElementById('questionForm');
const questionInput = document.getElementById('questionInput');
const submitBtn = document.getElementById('submitBtn');
const loadingSection = document.getElementById('loadingSection');
const errorSection = document.getElementById('errorSection');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');
const retryBtn = document.getElementById('retryBtn');
const clearBtn = document.getElementById('clearBtn');
const questionDisplay = document.getElementById('questionDisplay');
const answerDisplay = document.getElementById('answerDisplay');
const sparqlDisplay = document.getElementById('sparqlDisplay');
const rawDisplay = document.getElementById('rawDisplay');
const exampleButtons = document.querySelectorAll('.example-btn');

// State
let currentQuestion = '';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Attach event listeners
    questionForm.addEventListener('submit', handleSubmit);
    retryBtn.addEventListener('click', handleRetry);
    clearBtn.addEventListener('click', handleClear);

    // Attach example button listeners
    exampleButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const question = e.target.dataset.question;
            questionInput.value = question;
            questionInput.focus();
        });
    });

    // Check server health
    checkServerHealth();
});

// Check if the server is running
async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data = await response.json();
        console.log('Server health:', data);
    } catch (error) {
        console.warn('Server may not be running:', error);
    }
}

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();

    console.log('[DEBUG] Form submitted!');

    const question = questionInput.value.trim();
    console.log('[DEBUG] Question:', question);

    if (!question) {
        console.log('[DEBUG] Question is empty, showing error');
        showError('Please enter a question');
        return;
    }

    currentQuestion = question;

    console.log('[DEBUG] Hiding sections...');
    // Hide all sections
    hideAllSections();

    console.log('[DEBUG] Showing loading...');
    // Show loading
    showLoading();

    // Animate loading steps
    animateLoadingSteps();

    try {
        console.log('[DEBUG] Sending request to:', `${API_BASE_URL}/api/ask`);
        console.log('[DEBUG] Request body:', { question });

        const response = await fetch(`${API_BASE_URL}/api/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question }),
        });

        console.log('[DEBUG] Response status:', response.status);
        console.log('[DEBUG] Response ok:', response.ok);

        if (!response.ok) {
            const errorData = await response.json();
            console.log('[DEBUG] Error data:', errorData);
            throw new Error(errorData.error || 'Failed to process question');
        }

        const data = await response.json();
        console.log('[DEBUG] Received data:', data);

        // Display results
        displayResults(data);

    } catch (error) {
        console.error('[DEBUG] Error caught:', error);
        showError(error.message);
    }
}

// Show loading state
function showLoading() {
    loadingSection.style.display = 'block';
    submitBtn.disabled = true;
}

// Animate loading steps
function animateLoadingSteps() {
    const steps = document.querySelectorAll('.loading-step');
    let currentStep = 0;

    const interval = setInterval(() => {
        if (currentStep > 0) {
            steps[currentStep - 1].classList.remove('active');
        }
        if (currentStep < steps.length) {
            steps[currentStep].classList.add('active');
            currentStep++;
        } else {
            clearInterval(interval);
            // Loop back to first step
            setTimeout(() => {
                steps.forEach(step => step.classList.remove('active'));
                steps[0].classList.add('active');
                currentStep = 1;
            }, 500);
        }
    }, 1500);

    // Store interval ID for cleanup
    loadingSection.dataset.intervalId = interval;
}

// Display results
function displayResults(data) {
    // Clear loading interval
    if (loadingSection.dataset.intervalId) {
        clearInterval(parseInt(loadingSection.dataset.intervalId));
    }

    hideAllSections();

    // Populate result fields
    questionDisplay.textContent = data.question;
    answerDisplay.textContent = data.answer;
    sparqlDisplay.textContent = data.sparql_query || 'N/A';
    rawDisplay.textContent = data.raw_results || 'N/A';

    // Show results
    resultsSection.style.display = 'block';

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    // Re-enable submit button
    submitBtn.disabled = false;
}

// Show error
function showError(message) {
    // Clear loading interval
    if (loadingSection.dataset.intervalId) {
        clearInterval(parseInt(loadingSection.dataset.intervalId));
    }

    hideAllSections();

    errorMessage.textContent = message;
    errorSection.style.display = 'block';

    // Re-enable submit button
    submitBtn.disabled = false;
}

// Hide all sections
function hideAllSections() {
    loadingSection.style.display = 'none';
    errorSection.style.display = 'none';
    resultsSection.style.display = 'none';
}

// Handle retry
function handleRetry() {
    hideAllSections();
    questionInput.focus();
}

// Handle clear (new question)
function handleClear() {
    hideAllSections();
    questionInput.value = '';
    questionInput.focus();

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K to focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        questionInput.focus();
    }

    // Escape to clear
    if (e.key === 'Escape') {
        handleClear();
    }
});

// Add smooth reveal animation when results appear
function addRevealAnimation() {
    const cards = document.querySelectorAll('.answer-card, .sparql-details, .raw-details');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';

        setTimeout(() => {
            card.style.transition = 'all 0.5s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
}

// Call animation when results are displayed
const originalDisplayResults = displayResults;
displayResults = function (data) {
    originalDisplayResults(data);
    addRevealAnimation();
};
