const QUESTIONS_PER_USER = 12;
const MANIFEST_PATH = 'data/manifest.json';
const MIN_WAIT_MS = 10000;
const DEFAULT_PLAYBACK_RATE = 1.0;
const POOLTOOL_PLAYBACK_RATE = 0.5;
const COMPLETION_CODE = 'C6U7F00Y';
const COMPLETION_URL = `https://app.prolific.com/submissions/complete?cc=${encodeURIComponent(COMPLETION_CODE)}`;

const SUMMARY_VARIANTS = [
  { key: 'image_only', label: 'Image summary' },
  { key: 'supervised', label: 'Supervised summary' },
  { key: 'unsupervised', label: 'Unsupervised summary' },
];

const ENVIRONMENT_GOALS = {
  phyre: 'PHYRE is a 2D physics puzzle environment. The goal is to place a red ball so that the green and blue objects touch. You can pause the video and drag the progress bar to move backward or forward through the simulation before rating the summary.',
  iphyre: 'IPHYRE is a 2D physics puzzle environment. The goal is to get the red ball to fall into the abyss. You can pause the video and drag the progress bar to move backward or forward through the simulation before rating the summary.',
  pooltool: 'PoolTool is a billiards environment. The goal is to pocket coloured balls. You can pause the video and drag the progress bar to move backward or forward through the simulation before rating the summary.',
};

const query = new URLSearchParams(window.location.search);
const appBasePath = window.location.pathname.startsWith('/survey/') || window.location.pathname === '/survey' ? '/survey' : '';
const SAVE_API_PATH = `${appBasePath}/api/save`;
const prolificPid = String(query.get('PROLIFIC_PID') || '').trim();
const studyId = String(query.get('STUDY_ID') || '').trim();
const sessionId = String(query.get('SESSION_ID') || '').trim();
const fallbackUserId = Math.random().toString(36).slice(2, 11);
const participantId = prolificPid || fallbackUserId;
const rngSeed = hashStringToSeed(participantId);
const rng = createMulberry32(rngSeed);

const jsPsych = initJsPsych({
  on_finish: function () {
    if (prolificPid) {
      window.location.href = COMPLETION_URL;
    }
  },
});

function hashStringToSeed(str) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function createMulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t += 0x6D2B79F5;
    let x = t;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffleWithRng(array, randomFn) {
  const copy = [...array];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(randomFn() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    keepalive: true,
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

async function saveData(data) {
  const payload = {
    ...data,
    participant_id: participantId,
    prolific_pid: prolificPid || null,
    study_id: studyId || null,
    session_id: sessionId || null,
    PROLIFIC_PID: prolificPid || null,
    STUDY_ID: studyId || null,
    SESSION_ID: sessionId || null,
    prolific_params: {
      prolific_pid: prolificPid || null,
      study_id: studyId || null,
      session_id: sessionId || null,
    },
    rng_seed: rngSeed,
  };
  try {
    await postJson(SAVE_API_PATH, payload);
  } catch (error) {
    console.error('Error saving data:', error);
  }
}

function showError(message) {
  const container = document.querySelector('#jspsych-target') || document.body;
  container.innerHTML = `<div class="error-message">${escapeHtml(message)}</div>`;
}

async function loadManifest() {
  const response = await fetch(MANIFEST_PATH, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error('Unable to load study manifest.');
  }
  return response.json();
}

function buildVariantSchedule(count, randomFn) {
  const schedule = [];
  while (schedule.length < count) {
    schedule.push(...SUMMARY_VARIANTS.map((item) => item.key));
  }
  return shuffleWithRng(schedule.slice(0, count), randomFn);
}

function buildAssignments(manifest, randomFn) {
  const items = Array.isArray(manifest.items) ? manifest.items : [];
  if (items.length < QUESTIONS_PER_USER) {
    throw new Error(`Not enough complete items for the study. Need ${QUESTIONS_PER_USER}, found ${items.length}.`);
  }

  const selectedItems = shuffleWithRng(items, randomFn).slice(0, QUESTIONS_PER_USER);
  const variantSchedule = buildVariantSchedule(QUESTIONS_PER_USER, randomFn);

  return selectedItems.map((item, index) => {
    const variantKey = variantSchedule[index];
    const summary = item.summaries?.[variantKey];
    if (!summary || !String(summary.summary_text || '').trim()) {
      throw new Error(`Missing ${variantKey} summary for item ${item.item_id}.`);
    }
    return {
      trialNumber: index + 1,
      itemId: item.item_id,
      envId: item.env_id,
      taskId: item.task_id,
      sourceRunId: item.source_run_id,
      videoPath: item.video_path,
      initialImagePath: item.initial_image_path,
      summaryVariant: variantKey,
      summaryText: String(summary.summary_text || ''),
      annotationPatternCount: summary.annotation_pattern_count ?? null,
      environmentGoal: ENVIRONMENT_GOALS[item.env_id] || 'Goal: judge whether the described outcome matches what happens in the video.',
      playbackRate: item.env_id === 'pooltool' ? POOLTOOL_PLAYBACK_RATE : DEFAULT_PLAYBACK_RATE,
    };
  });
}

const welcome = {
  type: jsPsychHtmlButtonResponse,
  stimulus: `
    <div class="welcome-container">
      <h1 class="welcome-title">Summary Accuracy Study</h1>
      <p class="welcome-description">You will answer 12 questions.</p>
      <p class="welcome-description">For each question, watch a short video, read the summary, and rate how well the summary matches the video.</p>
      <p class="welcome-description">Use a score from 1 to 7, where 1 means very inaccurate and 7 means very accurate.</p>
      <p class="welcome-description">If you need context, hover over the yellow simulation explainer icon.</p>
    </div>
  `,
  choices: ['Start'],
  button_html: '<button class="jspsych-btn">%choice%</button>',
};

function createRatingTrial(assignment) {
  let trialStartTime = null;
  let videoDurationMs = null;
  let playbackRateIntervalId = null;
  let ratingCountdownIntervalId = null;
  const videoId = `video-${assignment.trialNumber}`;
  const timerId = `timer-${assignment.trialNumber}`;
  const choices = ['1', '2', '3', '4', '5', '6', '7'];

  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: () => {
      const summaryHtml = escapeHtml(assignment.summaryText).replace(/\n/g, '<br>');
      return `
        <div class="trial-container">
          <div class="trial-header">
            <div>Question ${assignment.trialNumber} of ${QUESTIONS_PER_USER}</div>
            <div class="env-help-tooltip">
              <span class="env-help-label">Simulation explainer</span>
              <span class="env-help-icon">?</span>
              <span class="env-help-text">${escapeHtml(assignment.environmentGoal)}</span>
            </div>
          </div>
          <div class="video-summary-grid">
            <div class="video-wrapper ${assignment.envId === 'pooltool' ? 'video-wrapper-pooltool' : ''}">
              <video id="${videoId}" class="summary-video ${assignment.envId === 'pooltool' ? 'summary-video-pooltool' : ''}" preload="auto" controls loop playsinline>
                <source src="${escapeHtml(assignment.videoPath)}" type="video/mp4">
                Your browser does not support the video tag.
              </video>
            </div>
            <div class="summary-wrapper">
              <h3>Summary</h3>
              <div class="summary-box">${summaryHtml}</div>
            </div>
          </div>
          <div id="${timerId}" class="timer-message">Please wait 10 seconds before rating the summary.</div>
        </div>
      `;
    },
    choices,
    button_html: '<button class="jspsych-btn">%choice%</button>',
    prompt: `
      <div class="rating-prompt">
        <span>How accurate is this summary relative to the video?</span>
        <span class="help-tooltip">
          <span class="help-icon">?</span>
          <span class="help-text">
            Accurate summaries should match the main events, objects, sequence, and outcome shown in the video without inventing details.
          </span>
        </span>
      </div>
    `,
    on_load: () => {
      trialStartTime = Date.now();
      const buttons = document.querySelectorAll('.jspsych-btn');
      buttons.forEach((button) => {
        button.disabled = true;
      });
      const timerEl = document.getElementById(timerId);
      const video = document.getElementById(videoId);
      let maxWatched = 0;
      const updateTimerText = (remainingMs) => {
        if (!timerEl) {
          return;
        }
        const remainingSeconds = Math.max(0, Math.ceil(remainingMs / 1000));
        if (remainingSeconds > 0) {
          timerEl.textContent = `Please wait ${remainingSeconds} second${remainingSeconds === 1 ? '' : 's'} before rating the summary.`;
        } else {
          timerEl.textContent = 'You may now rate the summary.';
        }
      };
      const enforcePlaybackRate = () => {
        video.defaultPlaybackRate = assignment.playbackRate;
        if (video.playbackRate !== assignment.playbackRate) {
          video.playbackRate = assignment.playbackRate;
        }
      };
      updateTimerText(MIN_WAIT_MS);
      enforcePlaybackRate();
      video.addEventListener('loadedmetadata', () => {
        videoDurationMs = video.duration * 1000;
        enforcePlaybackRate();
      });
      video.addEventListener('loadeddata', enforcePlaybackRate);
      video.addEventListener('canplay', enforcePlaybackRate);
      video.addEventListener('play', () => {
        enforcePlaybackRate();
      });
      video.addEventListener('ratechange', enforcePlaybackRate);
      video.addEventListener('timeupdate', () => {
        enforcePlaybackRate();
        if (video.currentTime > maxWatched) {
          maxWatched = video.currentTime;
        }
      });
      video.addEventListener('seeking', () => {
        enforcePlaybackRate();
        if (video.currentTime > maxWatched + 0.5) {
          video.currentTime = maxWatched;
        }
      });
      playbackRateIntervalId = window.setInterval(enforcePlaybackRate, 250);
      if (MIN_WAIT_MS > 0) {
        const countdownStart = Date.now();
        ratingCountdownIntervalId = window.setInterval(() => {
          const elapsedMs = Date.now() - countdownStart;
          updateTimerText(MIN_WAIT_MS - elapsedMs);
        }, 250);
      }
      window.setTimeout(() => {
        buttons.forEach((button) => {
          button.disabled = false;
        });
        if (ratingCountdownIntervalId != null) {
          window.clearInterval(ratingCountdownIntervalId);
          ratingCountdownIntervalId = null;
        }
        updateTimerText(0);
      }, MIN_WAIT_MS);
      const playbackPromise = video.play();
      if (playbackPromise && playbackPromise.catch) {
        playbackPromise.catch(() => {
          if (timerEl) {
            const remainingSeconds = Math.max(0, Math.ceil(MIN_WAIT_MS / 1000));
            timerEl.textContent = `Press play to watch the video. Rating will still unlock after ${remainingSeconds} second${remainingSeconds === 1 ? '' : 's'}.`;
          }
        });
      }
    },
    on_finish: (data) => {
      if (playbackRateIntervalId != null) {
        window.clearInterval(playbackRateIntervalId);
      }
      if (ratingCountdownIntervalId != null) {
        window.clearInterval(ratingCountdownIntervalId);
      }
      const responseTimeMs = Date.now() - trialStartTime;
      const rating = data.response != null ? parseInt(choices[data.response], 10) : null;
      saveData({
        task: 'rating',
        question_index: assignment.trialNumber,
        env_id: assignment.envId,
        item_id: assignment.itemId,
        task_id: assignment.taskId,
        source_run_id: assignment.sourceRunId,
        summary_variant: assignment.summaryVariant,
        summary_label: assignment.summaryLabel,
        summary_text: assignment.summaryText,
        annotation_pattern_count: assignment.annotationPatternCount,
        video_path: assignment.videoPath,
        initial_image_path: assignment.initialImagePath,
        playback_rate: assignment.playbackRate,
        rating,
        response_time_ms: responseTimeMs,
        jspsych_rt_ms: data.rt,
        video_duration_ms: videoDurationMs,
      });
    },
  };
}

const commentBlock = {
  type: jsPsychSurveyText,
  button_label: 'Submit feedback / Skip',
  questions: [
    {
      prompt: 'Optional feedback: share any comments about the summaries, videos, or study experience. Leave this blank and press the button below if you want to skip.',
      placeholder: 'Type your comments here, or leave blank to skip...',
      rows: 4,
      required: false,
    },
  ],
  on_finish: function (data) {
    saveData({
      task: 'comment',
      comment: data.response.Q0,
    });
    saveData({
      task: 'session_complete',
      completed: true,
    });
    window.setTimeout(() => {
      window.location.href = COMPLETION_URL;
    }, 150);
  },
};

async function startExperiment() {
  try {
    const manifest = await loadManifest();
    const assignments = buildAssignments(manifest, rng);
    await saveData({
      task: 'session_start',
      question_count: assignments.length,
      selected_item_ids: assignments.map((item) => item.itemId),
      selected_summary_variants: assignments.map((item) => item.summaryVariant),
    });

    const preload = {
      type: jsPsychPreload,
      video: assignments.map((item) => item.videoPath),
      show_progress_bar: true,
      message: 'Loading videos...',
    };

    const timeline = [welcome, preload];
    assignments.forEach((assignment) => {
      timeline.push(createRatingTrial(assignment));
    });
    timeline.push(commentBlock);
    jsPsych.run(timeline);
  } catch (error) {
    console.error(error);
    showError(error.message || 'An unexpected error occurred while loading the study.');
  }
}

startExperiment();
