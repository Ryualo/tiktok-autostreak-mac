# TikTok DM Auto-Streak Engine (macOS)

A low-level, hardware-accelerated automation tool for macOS that sends scheduled or bulk direct messages on TikTok Web. It uses Apple's native Vision OCR and OpenCV to dynamically read the chat sidebar, verify recipients, and send messages with strict hardware-level safety controls.

Built specifically to handle TikTok's edge cases, including invisible/blank names, 1-character display names, and tiny unique symbols.

## 🌟 Key Features

* **Strict Vision OCR Recognition:** Reads Display Names directly from the screen. Normalizes OCR confusions (e.g., `l` -> `i`, `zarı` -> `zari`) and dynamically handles truncated text.
* **Edge-Case Resolution:**
  * **Emoji-Only Blank Rows:** Calculates expected pixel heights to detect and message blank 8-space usernames even when the preview only contains an emoji (e.g., `🔥`).
  * **Tiny Symbol Detection:** Maps symbols like `π` or `.` using tight threshold pixel-brightness bounding boxes, even when Vision OCR completely misses them.
* **Smart 1-Day Interval Tracking:** Maintains a local `sent_history.json` to ensure you never accidentally double-message someone on the same calendar day.
* **Mandatory Header Verification:** Reads the top DM Header (`@username`) before typing to ensure the chat perfectly matches the intended recipient.
* **Hardware-Level Failsafes:** 
  * **Physical Mouse Tracking:** Tracks expected programmatic cursor movements. If you manually move your mouse more than 12px, the script triggers an instant Emergency Stop.
  * **Hotkeys:** Instantly abort by pressing `ESC` or `F12`. Pause/Resume using `F8`.

## ⚙️ Prerequisites

* **macOS** (Strictly required. Uses native Apple Quartz and Vision frameworks).
* **Python 3.8+**
* Google Chrome (or configured browser with TikTok logged in).

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Ryualo/tiktok-autostreak-mac.git](https://github.com/Ryualo/tiktok-autostreak-mac.git)
   cd tiktok-autostreak-mac
