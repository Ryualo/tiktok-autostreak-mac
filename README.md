# macOS TikTok DM Automation Engine

A Python-based automation tool for macOS that sends scheduled or bulk direct messages on TikTok Web. It uses Apple's native Vision OCR and OpenCV to dynamically read the chat sidebar, verify recipients, and send messages with strict safety controls.

## Features
* **Vision OCR Recognition:** Reads Display Names and timestamps directly from the screen, handling short names, unique symbols (like `π`), and truncated text.
* **Smart Interval Tracking:** Maintains a local `sent_history.json` to ensure you never accidentally double-message someone on the same calendar day.
* **Strict Verification:** Reads the DM Header (`@username`) before typing to ensure the chat matches the intended recipient. 
* **Hardware-Level Safety:** Includes programmatic cursor tracking. If you physically move your mouse, or press `ESC`/`F12`, the script instantly aborts. `F8` pauses execution.

## Prerequisites
* **macOS** (Required for Quartz and Apple Vision OCR)
* Python 3.8+
* Google Chrome (or configured browser)

## Installation

1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/tiktok-dm-automation.git](https://github.com/YOUR_USERNAME/tiktok-dm-automation.git)
   cd tiktok-dm-automation
