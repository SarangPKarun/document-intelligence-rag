document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatHistory = document.getElementById('chat-history');
    const clearChatBtn = document.getElementById('clear-chat');

    // --- File Upload Logic ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleUpload(fileInput.files[0]);
        }
    });

    async function handleUpload(file) {
        showStatus('Uploading ' + file.name + '...', 'info');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/ingest', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                showStatus('Success: ' + file.name + ' ingested!', 'success');
            } else {
                throw new Error('Upload failed');
            }
        } catch (error) {
            console.error(error);
            showStatus('Error uploading file.', 'error');
        }
    }

    function showStatus(msg, type) {
        uploadStatus.textContent = msg;
        uploadStatus.className = 'status-message visible ' + type;

        // Auto-hide success messages after 3s
        if (type === 'success') {
            setTimeout(() => {
                uploadStatus.classList.add('hidden');
                uploadStatus.classList.remove('visible');
            }, 3000);
        } else {
            uploadStatus.classList.remove('hidden');
        }
    }

    // --- Chat Logic ---
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = userInput.value.trim();
        if (!message) return;

        // Add User Message
        addMessage(message, 'user');
        userInput.value = '';

        // Show Loading
        const loadingId = addLoadingMessage();

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: message })
            });

            const data = await response.json();

            // Remove Loading
            removeMessage(loadingId);

            if (response.ok) {
                addMessage(data.answer, 'ai', data.context);
            } else {
                addMessage('Sorry, I encountered an error.', 'ai');
            }

        } catch (error) {
            removeMessage(loadingId);
            addMessage('Network error. Please try again.', 'ai');
            console.error(error);
        }
    });

    clearChatBtn.addEventListener('click', () => {
        chatHistory.innerHTML = `
            <div class="message ai-message">
                <div class="avatar">AI</div>
                <div class="bubble">
                    Chat cleared. How can I help you now?
                </div>
            </div>`;
    });

    function addMessage(text, sender, context = []) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = sender === 'ai' ? 'AI' : 'You';

        const bubble = document.createElement('div');
        bubble.className = 'bubble';

        // Use simple markdown-like parsing or just plain text
        // For security, strictly textcontent usually, but here simplicity prevails
        bubble.innerText = text;

        if (context && context.length > 0) {
            const contextDiv = document.createElement('div');
            contextDiv.className = 'context-box';
            contextDiv.innerHTML = `<span class="context-title">Context used:</span>${context[0]}`; // Showing first context snippet for brevity
            bubble.appendChild(contextDiv);
        }

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(bubble);

        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function addLoadingMessage() {
        const id = 'loading-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ai-message';
        msgDiv.id = id;

        msgDiv.innerHTML = `
            <div class="avatar">AI</div>
            <div class="bubble">
                <span class="typing-indicator">Thinking...</span>
            </div>
        `;

        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return id;
    }

    function removeMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }
});
