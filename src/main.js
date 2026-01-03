const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const { save } = window.__TAURI__.dialog;

const state = {
  unlisten: null,
  telemetryEnabled: false,
  currentPreviewFile: null,
};

function appendLog(line) {
  const consoleOutput = document.querySelector("#console-output");
  const currentText = consoleOutput.textContent;
  if (currentText.length > 50000) {
    consoleOutput.textContent = currentText.substring(currentText.length - 50000);
  }
  consoleOutput.textContent += `${line}\n`;
  consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

function setStatus(stateValue, text) {
  const statusLine = document.querySelector(".status-line");
  const statusText = document.querySelector("#status-text");
  if (statusLine) statusLine.dataset.state = stateValue;
  if (statusText) {
    statusText.dataset.state = stateValue;
    if (text) statusText.textContent = text;
  }
}

async function runExtractor(e) {
  e.preventDefault();

  const urlInput = document.querySelector('#input-url');
  const runButton = document.querySelector('#run-button');
  const consoleOutput = document.querySelector('#console-output');
  const modelInput = document.querySelector('input[name="whisper-model"]:checked');
  const model = modelInput ? modelInput.value : 'small';
  const langInput = document.querySelector('input[name="whisper-language"]:checked');
  const language = langInput ? langInput.value : 'auto';

  if (!urlInput.value.trim()) {
    appendLog('>> Provide a video URL first...');
    return;
  }

  runButton.disabled = true;
  setStatus('running', 'DEPLOYING PAYLOAD...');
  consoleOutput.textContent = '>> INITIALIZING NODE REACTOR...\n>> SPINNING UP WHISPER ENGINES...\n';

  // Always ensure telemetry listener is active during a run
  if (state.enableTelemetry) {
    await state.enableTelemetry();
  }

  try {
    await invoke('run_cli', {
      youtubeUrl: urlInput.value.trim(),
      model,
      language,
    });
    setStatus('done', 'MISSION COMPLETE');
    appendLog('>> PROCESS FINISHED.');
  } catch (error) {
    setStatus('error', 'MISSION FAILED');
    appendLog(`>> ERROR: ${error}`);
  } finally {
    runButton.disabled = false;
    if (state.unlisten) {
      state.unlisten();
      state.unlisten = null;
    }
  }
}

async function loadFileList() {
  const fileList = document.querySelector("#file-list");
  try {
    const files = await invoke("get_output_files");
    fileList.innerHTML = "";

    if (files.length === 0) {
      fileList.innerHTML = '<div class="empty-state">NO DATA CAPTURED</div>';
      return;
    }

    files.forEach(file => {
      const item = document.createElement("div");
      item.className = "file-item";
      item.innerHTML = `
        <span class="name">${file}</span>
        <span class="status">READY</span>
        <button class="delete-btn" aria-label="Delete file">✕</button>
        <button class="chip-btn save-btn" aria-label="Save file">SAVE</button>
      `;

      const deleteBtn = item.querySelector(".delete-btn");
      const saveBtn = item.querySelector(".save-btn");

      deleteBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        try {
          await invoke("delete_output_file", { filename: file });
          appendLog(`>> Deleted ${file}`);
          await loadFileList();
        } catch (err) {
          appendLog(`>> ERROR deleting ${file}: ${err}`);
          alert("DELETE FAILED: " + err);
        }
      });

      saveBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        saveFile(file);
      });

      item.onclick = () => showPreview(file);
      fileList.appendChild(item);
    });
  } catch (error) {
    console.error("Failed to load files:", error);
  }
}

async function saveFile(filename) {
  try {
    const content = await invoke("read_output_file", { filename });
    const path = await save({
      defaultPath: filename,
      filters: [{
        name: 'Text Files',
        extensions: ['txt']
      }]
    });

    if (path) {
      await invoke("save_content_to_file", { filepath: path, content });
      alert("FILE SECURED: " + path);
    }
  } catch (error) {
    alert("SAVE FAILED: " + error);
  }
}

async function showPreview(filename) {
  try {
    const content = await invoke("read_output_file", { filename });
    const preview = document.querySelector("#preview-output");
    state.currentPreviewFile = filename;
    preview.textContent = content || "(empty file)";
  } catch (error) {
    appendLog(`>> ERROR loading preview: ${error}`);
  }
}

async function copyPreview() {
  const preview = document.querySelector("#preview-output");
  if (!preview || !preview.textContent) return;
  try {
    await navigator.clipboard.writeText(preview.textContent);
    appendLog(">> Preview copied to clipboard");
  } catch (error) {
    appendLog(`>> ERROR copying preview: ${error}`);
    alert("COPY FAILED: " + error);
  }
}

async function savePreview() {
  if (!state.currentPreviewFile) {
    alert("No file selected");
    return;
  }
  await saveFile(state.currentPreviewFile);
}

window.addEventListener("DOMContentLoaded", () => {
  document.querySelector("#hack-form").addEventListener("submit", async (e) => {
    await runExtractor(e);
    await loadFileList();
  });

  document.querySelector("#refresh-btn").addEventListener("click", loadFileList);
  const copyBtn = document.querySelector("#preview-copy");
  const saveBtn = document.querySelector("#preview-save");
  copyBtn.addEventListener("click", copyPreview);
  saveBtn.addEventListener("click", savePreview);

  const grid = document.querySelector("main.grid");
  const consolePanel = document.querySelector(".console-panel");
  const telemetryPanelToggle = document.querySelector("#telemetry-panel-toggle");
  const telemetryToggle = document.querySelector("#telemetry-toggle");

  const updateTelemetryUI = (enabled) => {
    if (telemetryToggle) {
      telemetryToggle.textContent = enabled ? "DISABLE" : "ENABLE";
      telemetryToggle.setAttribute("aria-pressed", enabled ? "true" : "false");
    }
    if (consolePanel) {
      consolePanel.classList.toggle("telemetry-active", enabled);
    }
  };

  state.enableTelemetry = async () => {
    if (state.telemetryEnabled) return;
    if (state.unlisten) {
      state.unlisten();
      state.unlisten = null;
    }
    state.unlisten = await listen("log-output", (event) => {
      appendLog(event.payload);
    });
    state.telemetryEnabled = true;
    updateTelemetryUI(true);
  };

  state.disableTelemetry = () => {
    if (state.unlisten) {
      state.unlisten();
      state.unlisten = null;
    }
    state.telemetryEnabled = false;
    updateTelemetryUI(false);
  };

  const setTelemetryPanelVisibility = (show) => {
    if (!grid) return;
    const isVisible = !!show;
    grid.classList.toggle("telemetry-hidden", !isVisible);
    if (telemetryPanelToggle) {
      telemetryPanelToggle.textContent = isVisible ? "HIDE LIVE TELEMETRY" : "SHOW LIVE TELEMETRY";
      telemetryPanelToggle.setAttribute("aria-pressed", isVisible ? "true" : "false");
    }
    if (!isVisible) {
      state.disableTelemetry();
    }
  };

  telemetryPanelToggle?.addEventListener("click", () => {
    const currentlyHidden = grid?.classList.contains("telemetry-hidden");
    setTelemetryPanelVisibility(currentlyHidden);
  });

  setTelemetryPanelVisibility(false);

  telemetryToggle.addEventListener("click", () => {
    if (state.telemetryEnabled) {
      state.disableTelemetry();
    } else {
      state.enableTelemetry();
    }
  });

  // Start listening immediately so logs are captured even if panel is hidden
  state.enableTelemetry();

  loadFileList();
  setStatus("idle", "IDLE");

  const urlInput = document.querySelector("#input-url");
  const urlLabel = document.querySelector("#input-url-label");
  const radioButtons = document.querySelectorAll('input[name="video-type"]');

  radioButtons.forEach(radio => {
    radio.addEventListener('change', (e) => {
      if (e.target.value === 'tiktok') {
        urlLabel.textContent = "TIKTOK URL";
        urlInput.placeholder = "https://www.tiktok.com/@user/video/...";
      } else {
        urlLabel.textContent = "YOUTUBE URL";
        urlInput.placeholder = "https://youtube.com/watch?v=...";
      }
      urlInput.value = "";
    });
  });
});
