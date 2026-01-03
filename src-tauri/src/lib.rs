use num_cpus;
use regex::Regex;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::{error::Error, fmt};
use tauri::{Emitter, Window};
use whisper_rs::{
    get_lang_str, FullParams, SamplingStrategy, SegmentCallbackData, WhisperContext,
    WhisperContextParameters,
};

const DEFAULT_MODEL: &str = "base";
const YT_DLP_URL: &str = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe";
const FFMPEG_ZIP_URL: &str = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip";

static JOB_RUNNING: AtomicBool = AtomicBool::new(false);
static CANCEL_FLAG: AtomicBool = AtomicBool::new(false);

struct AppPaths {
    base: PathBuf,
    output: PathBuf,
    temp: PathBuf,
    bin: PathBuf,
    models: PathBuf,
}

struct JobGuard;
impl Drop for JobGuard {
    fn drop(&mut self) {
        JOB_RUNNING.store(false, Ordering::SeqCst);
        CANCEL_FLAG.store(false, Ordering::SeqCst);
    }
}

#[derive(Debug)]
enum AppError {
    Network(String),
    YtDlp(String),
    AudioTooQuiet,
    ModelLoad(String),
    Whisper(String),
    Cancelled,
    Security(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Network(msg) => write!(f, "Network error: {msg}"),
            AppError::YtDlp(msg) => write!(f, "Download error: {msg}"),
            AppError::AudioTooQuiet => write!(f, "Audio is too quiet"),
            AppError::ModelLoad(msg) => write!(f, "Model load failed: {msg}"),
            AppError::Whisper(msg) => write!(f, "Transcription failed: {msg}"),
            AppError::Cancelled => write!(f, "Operation cancelled"),
            AppError::Security(msg) => write!(f, "Security error: {msg}"),
        }
    }
}

impl Error for AppError {}

fn log_line(window: &Window, line: impl AsRef<str>) {
    let _ = window.emit("log-output", line.as_ref());
}

// Append a message to a log file under temp/logs; never panic on failure.
fn write_log(path: &Path, msg: &str) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "{}", msg);
    }
}

fn app_paths() -> Result<AppPaths, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let base = exe
        .parent()
        .ok_or("Cannot resolve executable directory")?
        .to_path_buf();

    let output = base.join("output");
    let temp = base.join("temp");
    let bin = base.join("bin");
    let models = base.join("models");

    for dir in [&output, &temp, &bin, &models] {
        if !dir.exists() {
            fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
    }

    Ok(AppPaths {
        base,
        output,
        temp,
        bin,
        models,
    })
}

fn detect_platform(url: &str) -> Option<&'static str> {
    let lower = url.to_lowercase();
    if lower.contains("youtube.com") || lower.contains("youtu.be") {
        Some("youtube")
    } else if lower.contains("tiktok.com")
        || lower.contains("vt.tiktok.com")
        || lower.contains("vm.tiktok.com")
    {
        Some("tiktok")
    } else {
        None
    }
}

fn extract_video_id(url: &str) -> Option<String> {
    let patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/)([^&\n?#]+)",
        r"youtube\.com/embed/([^&\n?#]+)",
        r"youtube\.com/v/([^&\n?#]+)",
        r"youtube\.com/shorts/([^&\n?#]+)",
        r"tiktok\.com/@[\w.-]+/video/(\d+)",
        r"vm\.tiktok\.com/(\w+)",
        r"vt\.tiktok\.com/(\w+)",
    ];

    for pat in patterns {
        if let Ok(re) = Regex::new(pat) {
            if let Some(cap) = re.captures(url) {
                if let Some(m) = cap.get(1) {
                    return Some(m.as_str().to_string());
                }
            }
        }
    }
    None
}

fn decode_xml(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
}

fn get_video_title(video_id: &str) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct OEmbedResp {
        title: String,
    }
    let url = format!(
        "https://www.youtube.com/oembed?url=https://youtube.com/watch?v={video_id}&format=json"
    );
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .ok()?;
    let resp = client.get(url).send().ok()?;
    let data: OEmbedResp = resp.json().ok()?;
    Some(data.title)
}

fn extract_transcript(video_id: &str) -> Option<String> {
    let url = format!("https://www.youtube.com/api/timedtext?v={video_id}&lang=en");
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .ok()?;
    let resp = client.get(url).send().ok()?;
    let body = resp.text().ok()?;

    let re = Regex::new(r"<text[^>]*>([^<]+)</text>").ok()?;
    let mut parts = Vec::new();
    for cap in re.captures_iter(&body) {
        if let Some(mat) = cap.get(1) {
            parts.push(decode_xml(mat.as_str()));
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

fn sanitize_id(raw: &str) -> String {
    let re = Regex::new(r"[^\w\.-]").unwrap();
    re.replace_all(raw, "_").to_string()
}

fn download_file(url: &str, dest: &Path) -> Result<(), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(|e| e.to_string())?;
    let mut resp = client.get(url).send().map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("Download failed: {}", resp.status()));
    }
    let mut file = fs::File::create(dest).map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    resp.copy_to(&mut buf).map_err(|e| e.to_string())?;
    std::io::copy(&mut buf.as_slice(), &mut file).map_err(|e| e.to_string())?;
    Ok(())
}

fn ensure_yt_dlp(paths: &AppPaths) -> Result<PathBuf, String> {
    let target = paths.bin.join("yt-dlp.exe");
    if target.exists() {
        return Ok(target);
    }
    download_file(YT_DLP_URL, &target)?;
    Ok(target)
}

fn extract_ffmpeg(zip_path: &Path, bin_dir: &Path) -> Result<PathBuf, String> {
    let file = fs::File::open(zip_path).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).map_err(|e| e.to_string())?;
        let name = file.name().to_string();
        if name.ends_with("ffmpeg.exe") {
            let out_path = bin_dir.join("ffmpeg.exe");
            let mut out_file = fs::File::create(&out_path).map_err(|e| e.to_string())?;
            std::io::copy(&mut file, &mut out_file).map_err(|e| e.to_string())?;
            return Ok(out_path);
        }
    }
    Err("ffmpeg.exe not found in archive".to_string())
}

fn ensure_ffmpeg(paths: &AppPaths, window: &Window) -> Result<PathBuf, String> {
    let target = paths.bin.join("ffmpeg.exe");
    if target.exists() {
        return Ok(target);
    }
    log_line(
        window,
        ">> Downloading ffmpeg (~80MB) — this can take up to a minute on slow networks...",
    );
    let zip_path = paths.temp.join("ffmpeg.zip");
    download_file(FFMPEG_ZIP_URL, &zip_path)?;
    log_line(window, ">> Unzipping ffmpeg...");
    let extracted = extract_ffmpeg(&zip_path, &paths.bin)?;
    let _ = fs::remove_file(zip_path);
    Ok(extracted)
}

fn model_url_and_name(model: &str) -> (&'static str, &'static str) {
    match model {
        "tiny" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
            "ggml-tiny.bin",
        ),
        "base" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
            "ggml-base.bin",
        ),
        "small" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
            "ggml-small.bin",
        ),
        _ => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
            "ggml-base.bin",
        ),
    }
}

fn ensure_model(paths: &AppPaths, model: &str) -> Result<PathBuf, String> {
    let (url, filename) = model_url_and_name(model);
    let target = paths.models.join(filename);
    if target.exists() {
        return Ok(target);
    }
    download_file(url, &target)?;
    Ok(target)
}

fn download_audio(
    paths: &AppPaths,
    url: &str,
    video_id: &str,
    yt_dlp: &Path,
    ffmpeg: &Path,
) -> Result<PathBuf, String> {
    let temp_dir = &paths.temp;
    let safe_id = sanitize_id(video_id);
    let wav_path = temp_dir.join(format!("{safe_id}.wav"));
    if wav_path.exists() {
        let _ = fs::remove_file(&wav_path);
    }

    let mut cmd = Command::new(yt_dlp);
    cmd.args([
        url,
        "-f",
        "bestaudio/best",
        "-x",
        "--audio-format",
        "wav",
        "--postprocessor-args",
        "-ar 16000 -ac 1",
        "-o",
        temp_dir
            .join(format!("{safe_id}.%(ext)s"))
            .to_string_lossy()
            .as_ref(),
        "--ffmpeg-location",
        ffmpeg.to_string_lossy().as_ref(),
        "--quiet",
        "--no-warnings",
    ]);

    let output = cmd.output().map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }

    if wav_path.exists() {
        Ok(wav_path)
    } else {
        Err("Audio download did not produce WAV".to_string())
    }
}

fn read_wav_mono_f32(path: &Path) -> Result<Vec<f32>, String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();

    if spec.channels == 0 {
        return Err("Invalid audio: 0 channels".to_string());
    }
    if spec.sample_rate != 16_000 {
        return Err(format!(
            "Unexpected sample rate {} Hz (expected 16000)",
            spec.sample_rate
        ));
    }

    let samples: Result<Vec<f32>, _> = match spec.sample_format {
        hound::SampleFormat::Int => {
            if spec.bits_per_sample > 32 {
                return Err("Unsupported bit depth".to_string());
            }
            let scale = match spec.bits_per_sample {
                8 => 128f32,            // scale by actual depth so quiet audio isn't near-zero
                16 => i16::MAX as f32,
                24 => 8_388_608f32,     // 2^23
                32 => i32::MAX as f32,
                _ => i32::MAX as f32,
            };
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().collect(),
    };

    let mut samples = samples.map_err(|e| e.to_string())?;

    if spec.channels > 1 {
        let mut mono = Vec::with_capacity(samples.len() / spec.channels as usize);
        for chunk in samples.chunks(spec.channels as usize) {
            let sum: f32 = chunk.iter().take(spec.channels as usize).copied().sum();
            mono.push(sum / spec.channels as f32);
        }
        samples = mono;
    }

    if samples.is_empty() {
        return Err("Audio buffer is empty after normalization".to_string());
    }

    Ok(samples)
}

// Compute basic audio stats and reject clips that are too short or too quiet.
fn analyze_audio(audio: &[f32]) -> Result<(f32, f32, f32), String> {
    if audio.is_empty() {
        return Err("Audio buffer is empty".to_string());
    }
    let duration = audio.len() as f32 / 16_000.0;
    let mut peak = 0.0f32;
    let mut sum_sq = 0.0f64;
    for &s in audio {
        let abs = s.abs();
        if abs > peak {
            peak = abs;
        }
        sum_sq += (s as f64) * (s as f64);
    }
    let rms = (sum_sq / audio.len() as f64).sqrt() as f32;

    // Treat very short or effectively silent clips as invalid.
    if duration < 0.5 {
        return Err(format!("Audio too short ({duration:.2}s)"));
    }
    if rms < 1e-4 || peak < 1e-3 {
        return Err(format!("Audio too quiet (rms={rms:.6}, peak={peak:.6})"));
    }

    Ok((duration, rms, peak))
}

fn transcribe_with_whisper(
    model_path: &Path,
    audio: &[f32],
    language: Option<&str>,
    window: Option<&Window>,
) -> Result<String, String> {
    if CANCEL_FLAG.load(Ordering::SeqCst) {
        return Err("Cancelled".to_string());
    }

    let ctx = WhisperContext::new_with_params(
        model_path.to_string_lossy().as_ref(),
        WhisperContextParameters::default(),
    )
    .map_err(|e| format!("Failed to load model: {e}"))?;
    let mut state = ctx.create_state().map_err(|e| e.to_string())?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(num_cpus::get() as i32);
    params.set_translate(false);
    params.set_temperature(0.0);
    params.set_no_speech_thold(0.8); // Higher threshold to cut hallucinated speech.
    params.set_logprob_thold(-1.0);
    params.set_suppress_blank(true); // Drop blank tokens to reduce repeats.
    params.set_suppress_nst(true); // Filter non-speech tokens when available.

    let mut detected_lang_owned: Option<String> = None;
    let mut auto_detect = true;
    if let Some(lang) = language {
        if !lang.is_empty() && lang.to_lowercase() != "auto" {
            params.set_language(Some(lang));
            params.set_detect_language(false);
            auto_detect = false;
        }
    }
    if auto_detect {
        params.set_language(None);
        params.set_detect_language(true);
    }

    let last_streamed = Arc::new(Mutex::new(String::new()));

    if auto_detect {
        let threads = num_cpus::get();
        state
            .pcm_to_mel(audio, threads)
            .map_err(|e| format!("mel failed: {e}"))?;
        let mut chosen_lang = "vi".to_string();
        if let Ok((lang_id, probs)) = state.lang_detect(0, threads) {
            if let Some(short) = get_lang_str(lang_id) {
                let conf = probs.get(lang_id as usize).cloned().unwrap_or_default();
                if let Some(win) = window {
                    let _ = win.emit(
                        "log-output",
                        format!(
                            ">> Detected language: {short} (~{:.0}%); min=75%",
                            conf * 100.0
                        ),
                    );
                }
                if conf >= 0.75 {
                    chosen_lang = short.to_string();
                }
            }
        }
        detected_lang_owned = Some(chosen_lang);
        params.set_language(detected_lang_owned.as_deref());
        params.set_detect_language(false);
    }

    if let Some(win) = window {
        let win = win.clone();
        let last_streamed = last_streamed.clone();
        params.set_segment_callback_safe(move |data: SegmentCallbackData| {
            if CANCEL_FLAG.load(Ordering::SeqCst) {
                return;
            }
            let text = data.text.trim();
            if !text.is_empty() {
                if let Ok(mut last) = last_streamed.lock() {
                    if *last == text {
                        return;
                    }
                    *last = text.to_string();
                }
                let _ = win.emit("log-output", format!(">> [{}] {}", data.segment + 1, text));
            }
        });
    }

    state.full(params, audio).map_err(|e| e.to_string())?;

    let num_segments = state.full_n_segments();
    let mut result = String::new();
    let mut last_seg = String::new();
    for i in 0..num_segments {
        if let Some(seg) = state.get_segment(i) {
            if let Ok(seg_text) = seg.to_str() {
                let t = seg_text.trim();
                if !t.is_empty() && t != last_seg {
                    result.push_str(t);
                    result.push(' ');
                    last_seg = t.to_string();
                }
            }
        }
    }
    Ok(result.trim().to_string())
}

fn save_transcript(paths: &AppPaths, content: &str, filename: &str) -> Result<(), String> {
    let filepath = paths.output.join(filename);
    fs::write(filepath, content).map_err(|e| e.to_string())
}

fn process_video(
    window: &Window,
    url: &str,
    model: &str,
    language: Option<String>,
) -> Result<(), String> {
    if JOB_RUNNING
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("Job already running".to_string());
    }
    CANCEL_FLAG.store(false, Ordering::SeqCst);
    let _guard = JobGuard;

    if CANCEL_FLAG.load(Ordering::SeqCst) {
        return Err("Cancelled".to_string());
    }
    let paths = app_paths()?;
    let platform = detect_platform(url).ok_or("URL not supported")?;
    let video_id = extract_video_id(url).ok_or("Cannot extract video ID")?;
    let safe_id = sanitize_id(&video_id);
    let lang_choice = language.unwrap_or_else(|| "auto".to_string());
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let log_path = paths
        .temp
        .join("logs")
        .join(format!("{safe_id}_{ts}.log"));
    log_line(
        window,
        format!(">> Language: {}", if lang_choice == "auto" { "auto-detect" } else { lang_choice.as_str() }),
    );
    write_log(
        &log_path,
        &format!(
            "START url={url} platform={platform} video_id={video_id} model={model} lang={lang_choice}"
        ),
    );

    log_line(window, ">> Analyzing URL...");
    log_line(
        window,
        format!(">> Platform: {platform}, video: {video_id}"),
    );

    log_line(window, ">> Checking yt-dlp...");
    let yt_dlp = ensure_yt_dlp(&paths)?;
    log_line(window, format!(">> yt-dlp ready at {}", yt_dlp.display()));
    write_log(&log_path, &format!("yt-dlp ready: {}", yt_dlp.display()));

    log_line(window, ">> Checking ffmpeg...");
    let ffmpeg = ensure_ffmpeg(&paths, window)?;
    log_line(window, format!(">> ffmpeg ready at {}", ffmpeg.display()));
    write_log(&log_path, &format!("ffmpeg ready: {}", ffmpeg.display()));

    let mut transcript: Option<String> = None;

    // Skip YouTube timedtext to avoid stale/incorrect transcripts; always use Whisper.
    if platform == "youtube" {
        log_line(window, ">> Fetching video title...");
        if let Some(title) = get_video_title(&video_id) {
            log_line(window, format!(">> Title: {title}"));
        }
        log_line(window, ">> Using Whisper (timedtext skipped).");
    } else {
        log_line(window, ">> TikTok detected, using Whisper...");
    }

    if transcript.is_none() {
        log_line(window, ">> Downloading audio (yt-dlp + ffmpeg)...");
        if CANCEL_FLAG.load(Ordering::SeqCst) {
            return Err("Cancelled".to_string());
        }
        let wav_path = download_audio(&paths, url, &video_id, &yt_dlp, &ffmpeg)?;
        write_log(&log_path, &format!("audio downloaded: {}", wav_path.display()));
        log_line(
            window,
            ">> Audio ready, loading Whisper model (may take a bit if first time)...",
        );
        let model_path = ensure_model(&paths, model)?;
        log_line(
            window,
            format!(
                ">> Using model: {}",
                model_path.file_name().unwrap_or_default().to_string_lossy()
            ),
        );
        write_log(
            &log_path,
            &format!(
                "model selected: {}",
                model_path.file_name().unwrap_or_default().to_string_lossy()
            ),
        );
        log_line(window, ">> Reading WAV into memory...");
        if CANCEL_FLAG.load(Ordering::SeqCst) {
            write_log(&log_path, "CANCELLED");
            let _ = fs::remove_file(&wav_path);
            return Err("Cancelled".to_string());
        }
        let audio = read_wav_mono_f32(&wav_path)?;
        log_line(window, ">> Running Whisper inference...");
        if CANCEL_FLAG.load(Ordering::SeqCst) {
            write_log(&log_path, "CANCELLED");
            let _ = fs::remove_file(&wav_path);
            return Err("Cancelled".to_string());
        }
        match analyze_audio(&audio) {
            Ok((secs, rms, peak)) => {
                log_line(
                    window,
                    format!(">> Audio stats: {secs:.2}s rms={rms:.6} peak={peak:.6}"),
                );
                write_log(
                    &log_path,
                    &format!("audio stats: secs={secs:.2} rms={rms:.6} peak={peak:.6}"),
                );
            }
            Err(reason) => {
                log_line(window, format!(">> Audio rejected: {reason}"));
                let _ = fs::remove_file(&wav_path);
                write_log(&log_path, &format!("audio rejected: {reason}"));
                return Err(reason.to_string());
            }
        }
        let mut final_text = transcribe_with_whisper(
            &model_path,
            &audio,
            Some(lang_choice.as_str()),
            Some(window),
        )?;

        if final_text.trim().is_empty() && lang_choice == "auto" {
            log_line(
                window,
                ">> No text detected on auto; retrying with Vietnamese (vi)...",
            );
            write_log(&log_path, "auto-detect empty; retry vi");
            final_text = transcribe_with_whisper(&model_path, &audio, Some("vi"), Some(window))?;
        }
        if final_text.trim().is_empty() && lang_choice == "auto" {
            log_line(
                window,
                ">> Still empty on auto; retrying with English (en)...",
            );
            write_log(&log_path, "auto-detect still empty; retry en");
            final_text = transcribe_with_whisper(&model_path, &audio, Some("en"), Some(window))?;
        }

        transcript = Some(final_text);
        let _ = fs::remove_file(wav_path);
        log_line(window, ">> Whisper finished");
    }

    let transcript = transcript.ok_or("Failed to obtain transcript")?;
    if transcript.trim().is_empty() {
        return Err("No speech detected; try forcing language to VI or EN or use a higher-quality model.".to_string());
    }
    log_line(
        window,
        format!(">> Transcript length: {} chars", transcript.len()),
    );
    write_log(
        &log_path,
        &format!("transcript length: {}", transcript.len()),
    );

    let filename = format!("{safe_id}_{ts}.txt");
    save_transcript(&paths, &transcript, &filename)?;
    log_line(
        window,
        format!(">> Saved to output folder as {filename}"),
    );
    write_log(&log_path, &format!("saved file: {filename}"));
    Ok(())
}

#[tauri::command]
async fn run_cli(
    window: Window,
    youtube_url: String,
    model: Option<String>,
    language: Option<String>,
) -> Result<String, String> {
    let url = youtube_url.trim().to_string();
    let whisper_model = model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let lang_choice = language.unwrap_or_else(|| "auto".to_string());

    let handle = tauri::async_runtime::spawn_blocking(move || {
        process_video(&window, &url, &whisper_model, Some(lang_choice))
    });

    match handle.await {
        Ok(Ok(())) => Ok("Process finished successfully".to_string()),
        Ok(Err(err)) => Err(err),
        Err(e) => Err(format!("Thread join error: {e}")),
    }
}

#[tauri::command]
async fn get_output_files() -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    let paths = app_paths()?;

    if paths.output.exists() {
        if let Ok(entries) = fs::read_dir(&paths.output) {
            for entry in entries.flatten() {
                if let Ok(name) = entry.file_name().into_string() {
                    if name.ends_with(".txt") {
                        files.push(name);
                    }
                }
            }
        }
    }
    files.sort();
    files.reverse();
    Ok(files)
}

#[tauri::command]
async fn read_output_file(filename: String) -> Result<String, String> {
    let paths = app_paths()?;
    let file_path = paths.output.join(filename);

    if !file_path.exists() {
        return Err("File not found".to_string());
    }

    fs::read_to_string(file_path).map_err(|e| e.to_string())
}

#[tauri::command]
async fn save_content_to_file(filepath: String, content: String) -> Result<(), String> {
    let paths = app_paths().map_err(|e| e.to_string())?;
    let output_dir = paths.output;

    // Reject absolute paths to avoid path traversal and drive escapes.
    let user_path = PathBuf::from(filepath);
    if user_path.is_absolute() {
        return Err("Invalid path".to_string());
    }

    // Sanitize: disallow parent components and empty names.
    if user_path.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return Err("Invalid path".to_string());
    }
    let file_name = user_path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "Invalid filename".to_string())?;
    if file_name.is_empty() {
        return Err("Invalid filename".to_string());
    }
    if !file_name.to_lowercase().ends_with(".txt") {
        return Err("Invalid file type".to_string());
    }

    let dest = output_dir.join(file_name);
    fs::write(dest, content).map_err(|e| e.to_string())
}

#[tauri::command]
async fn delete_output_file(filename: String) -> Result<(), String> {
    let paths = app_paths()?;
    let file_path = paths.output.join(&filename);

    if !file_path.exists() {
        return Err("File not found".to_string());
    }
    if !filename.ends_with(".txt") {
        return Err("Invalid file type".to_string());
    }

    fs::remove_file(file_path).map_err(|e| e.to_string())
}

#[tauri::command]
async fn cancel_current_job() -> Result<(), String> {
    CANCEL_FLAG.store(true, Ordering::SeqCst);
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            run_cli,
            get_output_files,
            read_output_file,
            save_content_to_file,
            delete_output_file,
            cancel_current_job
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
