// ============================================================================
// TAB NAVIGATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;

            // Remove active class from all tabs and contents
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked tab and corresponding content
            btn.classList.add('active');
            document.getElementById(`${targetTab}-tab`).classList.add('active');
        });
    });

    // Initialize SPARQL and Embedding chat functionality
    initSPARQLTab();
    initEmbeddingChat();
});

// ============================================================================
// SPARQL TAB (Original functionality)
// ============================================================================

function initSPARQLTab() {
    const questionForm = document.getElementById('questionForm');
    const questionInput = document.getElementById('questionInput');
    const submitBtn = document.getElementById('submitBtn');
    const loadingSection = document.getElementById('loadingSection');
    const errorSection = document.getElementById('errorSection');
    const resultsSection = document.getElementById('resultsSection');
    const errorMessage = document.getElementById('errorMessage');
    const retryBtn = document.getElementById('retryBtn');
    const clearBtn = document.getElementById('clearBtn');

    // Example question buttons
    const exampleBtns = document.querySelectorAll('.example-btn');
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.dataset.question;
            questionInput.value = question;
            questionInput.focus();
        });
    });

    // Form submission
    questionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = questionInput.value.trim();

        if (!question) return;

        // Show loading, hide other sections
        showSection(loadingSection);
        hideSection(errorSection);
        hideSection(resultsSection);

        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to process question');
            }

            // Display results
            displaySPARQLResults(data);

        } catch (error) {
            console.error('Error:', error);
            errorMessage.textContent = error.message;
            showSection(errorSection);
            hideSection(loadingSection);
        }
    });

    // Retry button
    retryBtn.addEventListener('click', () => {
        hideSection(errorSection);
        questionForm.requestSubmit();
    });

    // Clear button
    clearBtn.addEventListener('click', () => {
        hideSection(resultsSection);
        questionInput.value = '';
        questionInput.focus();
    });
}

function displaySPARQLResults(data) {
    const { question, sparql_query, raw_results, answer } = data;

    // Display question
    document.getElementById('questionDisplay').textContent = question;

    // Display answer
    document.getElementById('answerDisplay').textContent = answer;

    // Display SPARQL query
    document.getElementById('sparqlDisplay').textContent = sparql_query;

    // Display raw results
    document.getElementById('rawDisplay').textContent = raw_results;

    // Show results section
    showSection(document.getElementById('resultsSection'));
    hideSection(document.getElementById('loadingSection'));
}

// ============================================================================
// EMBEDDING CHAT TAB
// ============================================================================

let chatHistory = [];

function initEmbeddingChat() {
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const chatSubmitBtn = document.getElementById('chatSubmitBtn');
    const chatMessages = document.getElementById('chatMessages');
    const clearChatBtn = document.getElementById('clearChatBtn');

    // Example buttons
    const chatExampleBtns = document.querySelectorAll('.chat-example-btn');
    chatExampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.dataset.question;
            chatInput.value = question;
            chatInput.focus();
        });
    });

    // Chat form submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = chatInput.value.trim();

        if (!question) return;

        // Add user message to chat
        addChatMessage('user', question);
        chatInput.value = '';

        // Disable input while processing
        chatInput.disabled = true;
        chatSubmitBtn.disabled = true;

        try {
            const response = await fetch('/api/ask_embedding', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question })
            });

            // Check if response is OK first
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server error (${response.status}): ${errorText}`);
            }

            // Try to parse JSON
            let data;
            try {
                data = await response.json();
            } catch (jsonError) {
                const responseText = await response.text();
                console.error('Failed to parse JSON. Response was:', responseText);
                throw new Error(`Invalid JSON response from server. Response: ${responseText.substring(0, 200)}`);
            }

            // Add bot response to chat with full context data
            addChatMessage('bot', data.answer, {
                num_entities: data.num_entities,
                num_triplets: data.num_triplets,
                processing_time: data.processing_time,
                retrieved_entities: data.retrieved_entities,
                retrieved_triplets: data.retrieved_triplets
            });

            // Store in history
            chatHistory.push({
                question: question,
                answer: data.answer,
                timestamp: new Date()
            });

        } catch (error) {
            console.error('Error:', error);
            addChatMessage('bot', `Sorry, I encountered an error: ${error.message}`, null, true);
        } finally {
            // Re-enable input
            chatInput.disabled = false;
            chatSubmitBtn.disabled = false;
            chatInput.focus();
        }
    });

    // Clear chat button
    clearChatBtn.addEventListener('click', () => {
        // Keep only the system message
        const systemMessage = chatMessages.querySelector('.system-message');
        chatMessages.innerHTML = '';
        if (systemMessage) {
            chatMessages.appendChild(systemMessage);
        }
        chatHistory = [];
        chatInput.focus();
    });
}

function addChatMessage(type, content, metadata = null, isError = false) {
    const chatMessages = document.getElementById('chatMessages');

    const messageDiv = document.createElement('div');
    messageDiv.className = `${type}-message`;

    const icon = document.createElement('div');
    icon.className = 'message-icon';
    icon.textContent = type === 'user' ? '👤' : '🤖';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const textP = document.createElement('p');
    textP.textContent = content;
    contentDiv.appendChild(textP);

    // Add metadata and retrieved context if present
    if (metadata && !isError) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-metadata';
        metaDiv.textContent = `Retrieved: ${metadata.num_entities} entities, ${metadata.num_triplets} triplets • ${metadata.processing_time.toFixed(2)}s`;
        contentDiv.appendChild(metaDiv);
    }

    messageDiv.appendChild(icon);
    messageDiv.appendChild(contentDiv);

    chatMessages.appendChild(messageDiv);

    // Add retrieved context details if available (from full data)
    if (metadata && metadata.retrieved_entities && metadata.retrieved_triplets) {
        addRetrievedContext(chatMessages, metadata.retrieved_entities, metadata.retrieved_triplets);
    }

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addRetrievedContext(chatMessages, entities, triplets) {
    const contextDiv = document.createElement('div');
    contextDiv.className = 'retrieved-context';

    // Entities section
    if (entities && entities.length > 0) {
        const entitiesDetails = document.createElement('details');
        entitiesDetails.className = 'context-details';

        const entitiesSummary = document.createElement('summary');
        entitiesSummary.className = 'context-summary';
        entitiesSummary.innerHTML = `
            <span class="summary-icon">🎬</span>
            <span>Retrieved Entities (${entities.length})</span>
            <span class="summary-arrow">▼</span>
        `;

        const entitiesContent = document.createElement('div');
        entitiesContent.className = 'context-content';

        entities.forEach((entity, index) => {
            const entityCard = document.createElement('div');
            entityCard.className = 'entity-card';
            entityCard.innerHTML = `
                <div class="entity-header">
                    <strong>${index + 1}. ${entity.title}</strong>
                    <span class="entity-type">${entity.entity_type}</span>
                </div>
                <div class="entity-text">${entity.text}</div>
            `;
            entitiesContent.appendChild(entityCard);
        });

        entitiesDetails.appendChild(entitiesSummary);
        entitiesDetails.appendChild(entitiesContent);
        contextDiv.appendChild(entitiesDetails);
    }

    // Triplets section
    if (triplets && triplets.length > 0) {
        const tripletsDetails = document.createElement('details');
        tripletsDetails.className = 'context-details';

        const tripletsSummary = document.createElement('summary');
        tripletsSummary.className = 'context-summary';
        tripletsSummary.innerHTML = `
            <span class="summary-icon">📊</span>
            <span>Retrieved Facts (${triplets.length})</span>
            <span class="summary-arrow">▼</span>
        `;

        const tripletsContent = document.createElement('div');
        tripletsContent.className = 'context-content';

        const tripletsList = document.createElement('ul');
        tripletsList.className = 'triplets-list';

        triplets.forEach(triplet => {
            const li = document.createElement('li');
            li.textContent = triplet.text;
            tripletsList.appendChild(li);
        });

        tripletsContent.appendChild(tripletsList);
        tripletsDetails.appendChild(tripletsSummary);
        tripletsDetails.appendChild(tripletsContent);
        contextDiv.appendChild(tripletsDetails);
    }

    chatMessages.appendChild(contextDiv);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function showSection(element) {
    element.style.display = 'block';
}

function hideSection(element) {
    element.style.display = 'none';
}
