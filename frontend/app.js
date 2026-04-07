const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const quickButtons = document.querySelectorAll('.quick-btn');
const presetFillButtons = document.querySelectorAll('.preset-fill-btn');
const apiKeyInput = document.getElementById('apiKeyInput');
const apiBaseInput = document.getElementById('apiBaseInput');
const modelInput = document.getElementById('modelInput');
const saveKeyBtn = document.getElementById('saveKeyBtn');
const keyHint = document.getElementById('keyHint');

const API_KEY_STORAGE_KEY = 'bupt_agent_api_key';
const API_BASE_STORAGE_KEY = 'bupt_agent_api_base';
const MODEL_STORAGE_KEY = 'bupt_agent_model';

// Generate a random user ID for the session to utilize memory checkpointer
const userId = 'user_' + Math.random().toString(36).substr(2, 9);

function addMessage(text, isUser = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${isUser ? 'user-msg' : 'system-msg'}`;
    
    let content = '';
    if (isUser) {
        content = `<div class="bubble">${escapeHTML(text)}</div>`;
    } else {
        content = `
            <div class="avatar">🤖</div>
            <div class="bubble glass-effect">${escapeHTML(text)}</div>
        `;
    }
    
    msgDiv.innerHTML = content;
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addTypingIndicator() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message system-msg';
    msgDiv.id = 'typingIndicator';
    msgDiv.innerHTML = `
        <div class="avatar">🤖</div>
        <div class="bubble glass-effect">
            <div class="typing-indicator">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>
    `;
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

function escapeHTML(str) {
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}

function getSavedApiKey() {
    return localStorage.getItem(API_KEY_STORAGE_KEY) || '';
}

function setSavedApiKey(value) {
    localStorage.setItem(API_KEY_STORAGE_KEY, value);
}

function getSavedApiBase() {
    return localStorage.getItem(API_BASE_STORAGE_KEY) || '';
}

function setSavedApiBase(value) {
    localStorage.setItem(API_BASE_STORAGE_KEY, value);
}

function getSavedModel() {
    return localStorage.getItem(MODEL_STORAGE_KEY) || '';
}

function setSavedModel(value) {
    localStorage.setItem(MODEL_STORAGE_KEY, value);
}

function updateKeyHint(text) {
    keyHint.textContent = text;
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // UI Updates
    addMessage(text, true);
    userInput.value = '';
    addTypingIndicator();
    sendBtn.disabled = true;

    try {
        const apiKey = (apiKeyInput.value || '').trim() || getSavedApiKey().trim();
        const apiBase = (apiBaseInput.value || '').trim() || getSavedApiBase().trim();
        const model = (modelInput.value || '').trim() || getSavedModel().trim();
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: userId,
                message: text,
                api_key: apiKey || null,
                api_base: apiBase || null,
                model: model || null
            })
        });

        const raw = await response.text();
        let data = null;
        try {
            data = raw ? JSON.parse(raw) : null;
        } catch (err) {
            data = null;
        }
        removeTypingIndicator();

        if (response.ok) {
            addMessage((data && data.reply) || '收到响应，但返回格式异常。');
        } else {
            const detail = (data && data.detail) || `接口错误（HTTP ${response.status}）`;
            addMessage('Oops! 出错了：' + detail);
        }
    } catch (error) {
        removeTypingIndicator();
        addMessage('网络请求失败，请确保后台 API 已正常运行。');
    } finally {
        sendBtn.disabled = false;
        userInput.focus();
    }
}

quickButtons.forEach((button) => {
    button.addEventListener('click', () => {
        userInput.value = button.dataset.prompt || '';
        userInput.focus();
    });
});

presetFillButtons.forEach((button) => {
    button.addEventListener('click', () => {
        userInput.value = button.dataset.prompt || '';
        userInput.focus();
    });
});

saveKeyBtn.addEventListener('click', () => {
    const value = (apiKeyInput.value || '').trim();
    if (!value) {
        localStorage.removeItem(API_KEY_STORAGE_KEY);
        localStorage.removeItem(API_BASE_STORAGE_KEY);
        localStorage.removeItem(MODEL_STORAGE_KEY);
        updateKeyHint('已清空浏览器中保存的 Key 与 Base URL。');
        return;
    }

    setSavedApiKey(value);
    setSavedApiBase((apiBaseInput.value || '').trim());
    setSavedModel((modelInput.value || '').trim());
    updateKeyHint('已保存到当前浏览器，可直接开始对话。');
});

const initialApiKey = getSavedApiKey();
if (initialApiKey) {
    apiKeyInput.value = initialApiKey;
    updateKeyHint('已检测到本地保存的 Key。');
}

const initialApiBase = getSavedApiBase();
if (initialApiBase) {
    apiBaseInput.value = initialApiBase;
}

const initialModel = getSavedModel();
if (initialModel) {
    modelInput.value = initialModel;
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});
