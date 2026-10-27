#!/usr/bin/env python3
"""
setup.py — Smooth 120Hz TikTok DM Region Calibrator with Live Tab Refresh
Targets Intel Macs + Google Chrome.

Upgrades:
  - Cached static canvas rendering: zero CPU resize during mouse drag (butter-smooth cropping).
  - Background threaded Vision OCR on mouse release so the UI never freezes.
  - Press [R] or [Space] (or click the top-right [REFRESH TIKTOK (R)] button) at any time
    to automatically switch Google Chrome to https://www.tiktok.com/messages, hide the
    setup window briefly, capture a fresh screenshot, and restore the setup window.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import AppKit
import cv2
import numpy as np
import Quartz
import Vision
from Foundation import NSURL

CONFIG_DEFAULT_PATH = Path("config.json")
NICKNAMES_DEFAULT_PATH = Path("nicknames.txt")
TIKTOK_MESSAGES_URL = "https://www.tiktok.com/messages"

DEFAULT_NICKNAMES_CONTENT = """# TikTok Recipients — One per line: Display Name | @username
# Editing nicknames/details in the app preserves permanent IDs & sent history.
F. | @madiksxort
Cayxl | @kaungkaung8908
daniel_the_duck | @daniel.duck08
. | @user07340968
K | @lucas_razen
WilliamDavidson | @williamdavidson488
M | @pcachu06
w | @wtfi.w
aungthett111 | @aungthett111
Shaun | @shaun1922
Boisvert 🎭 | @da.tommy7
chl. | @nikkkkkk_69
Kiwiii | @haerincoholic
\ufff4 \ufff4\ufff4 \ufff4 \ufff4\ufff4 \ufff4 \ufff4 \ufff4 \ufff4\ufff4\ufff4 \ufff4\ufff4 \ufff4\ufff4\ufff4 | @akatsuki7529
AParanoidAndroid | @slaxh7n
DoubleT | @autryyss
zari | @huachengdexiaobao
FuyuTheDumb | @fuyu_the_dumb
π | @paingpaing1123
Dalton Maverick | @rushmall
kise | @ilmsg_
"""

REGION_KEYS = ("dm_sidebar", "recipient_header", "message_input")

REGION_META: Dict[str, Dict[str, Any]] = {
    "dm_sidebar": {
        "label": "1. DM Sidebar (Conversation Rows)",
        "short": "DM Sidebar",
        "color_bgr": (255, 200, 0),
        "key_hint": "1",
    },
    "recipient_header": {
        "label": "2. Recipient Header (Name + @username)",
        "short": "Recipient Header",
        "color_bgr": (80, 220, 100),
        "key_hint": "2",
    },
    "message_input": {
        "label": "3. Message Input Box (Send a message...)",
        "short": "Message Input",
        "color_bgr": (60, 140, 255),
        "key_hint": "3",
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "version": "3.0.0",
    "browser_app": "Google Chrome",
    "nicknames_file": "nicknames.txt",
    "message": "🔥",
    "allow_duplicates": True,
    "verification_threshold": 0.80,
    "max_sidebar_scrolls": 3,
    "sidebar_row_height_logical": 72.0,
    "display_profile": {},
    "regions": {
        "dm_sidebar": None,
        "recipient_header": None,
        "message_input": None,
    },
}


@dataclass
class RecipientEntry:
    display_name: str
    username: str
    clean_display: str


@dataclass
class OCRBox:
    text: str
    confidence: float
    px_x: int
    px_y: int
    px_w: int
    px_h: int


def strip_invisible_and_emojis(text: str) -> str:
    out_chars: List[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("C") or ord(ch) in (
            0xFFF4, 0xFFFC, 0x200B, 0x200C, 0x200D, 0xFEFF, 0x3164, 0xFFA0
        ):
            continue
        if cat == "So":
            continue
        out_chars.append(ch)
    cleaned = "".join(out_chars).strip()
    return re.sub(r"\s+", " ", cleaned)


def ensure_and_parse_nicknames(path: Path) -> List[RecipientEntry]:
    if not path.exists():
        path.write_text(DEFAULT_NICKNAMES_CONTENT, encoding="utf-8")
        print(f"[INFO] Created default '{path}' with 21 recipients.")

    entries: List[RecipientEntry] = []
    seen_ids = set()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in raw_line:
            left, right = raw_line.split("|", 1)
            disp = left.strip()
            handle = right.strip()
        else:
            disp = line
            handle = line

        if not handle:
            continue
        if not handle.startswith("@"):
            handle = "@" + handle

        perm_id = handle.lower()
        if perm_id in seen_ids:
            continue
        seen_ids.add(perm_id)

        entries.append(
            RecipientEntry(
                display_name=disp,
                username=handle,
                clean_display=strip_invisible_and_emojis(disp),
            )
        )
    return entries


def get_logical_display_size() -> Tuple[int, int]:
    main_id = Quartz.CGMainDisplayID()
    return int(Quartz.CGDisplayPixelsWide(main_id)), int(
        Quartz.CGDisplayPixelsHigh(main_id)
    )


def focus_chrome_and_switch_to_tiktok(
    browser_app: str = "Google Chrome",
    switch_to_tiktok_tab: bool = True,
) -> None:
    """
    Activate Google Chrome. If `switch_to_tiktok_tab` is True, automatically
    switch to an existing tiktok.com/messages tab if one is open, or just bring
    Chrome's active tab to the front.
    """
    if switch_to_tiktok_tab:
        script = f"""
        tell application "{browser_app}"
            activate
            if (count of windows) > 0 then
                set activeURL to URL of active tab of front window
                if activeURL does not contain "tiktok.com/messages" then
                    repeat with w in windows
                        set tabIdx to 1
                        repeat with t in tabs of w
                            if (URL of t) contains "tiktok.com/messages" then
                                set active tab index of w to tabIdx
                                set index of w to 1
                                exit repeat
                            end if
                            set tabIdx to tabIdx + 1
                        end repeat
                    end repeat
                end if
            end if
        end tell
        """
    else:
        script = f'tell application "{browser_app}" to activate'

    subprocess.run(
        ["/usr/bin/osascript", "-e", script],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def capture_fullscreen_bgr() -> np.ndarray:
    fd, tmp_path = tempfile.mkstemp(prefix="tiktok_setup_cap_", suffix=".png")
    os.close(fd)
    try:
        res = subprocess.run(
            ["/usr/sbin/screencapture", "-x", "-m", tmp_path],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if res.returncode != 0 or not os.path.exists(tmp_path):
            raise RuntimeError(
                "screencapture failed. Grant Screen Recording permission to Terminal/IDE."
            )
        img = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise RuntimeError("Captured screenshot is empty.")
        return img
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def run_vision_ocr_on_bgr(crop_bgr: np.ndarray) -> List[OCRBox]:
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    img_h, img_w = crop_bgr.shape[:2]
    fd, tmp_path = tempfile.mkstemp(prefix="tiktok_setup_ocr_", suffix=".png")
    os.close(fd)

    results: List[OCRBox] = []
    try:
        cv2.imwrite(tmp_path, crop_bgr)
        file_url = NSURL.fileURLWithPath_(tmp_path)
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
            file_url, None
        )
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)
        if hasattr(request, "setMinimumTextHeight_"):
            request.setMinimumTextHeight_(0.0)

        ok, _err = handler.performRequests_error_([request], None)
        if not ok or request.results() is None:
            return []

        for obs in request.results():
            candidates = obs.topCandidates_(1)
            if not candidates or len(candidates) == 0:
                continue
            top = candidates[0]
            text_val = str(top.string()).strip()
            if not text_val:
                continue
            conf_val = float(top.confidence())
            bbox = obs.boundingBox()
            nx = float(bbox.origin.x)
            ny = float(bbox.origin.y)
            nw = float(bbox.size.width)
            nh = float(bbox.size.height)

            px_x = max(0, int(round(nx * img_w)))
            px_y = max(0, int(round((1.0 - ny - nh) * img_h)))
            px_w = max(1, int(round(nw * img_w)))
            px_h = max(1, int(round(nh * img_h)))

            results.append(
                OCRBox(
                    text=text_val,
                    confidence=conf_val,
                    px_x=px_x,
                    px_y=px_y,
                    px_w=px_w,
                    px_h=px_h,
                )
            )
        results.sort(key=lambda b: (b.px_y, b.px_x))
        return results
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def load_or_init_config(path: Path) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                for k, v in loaded.items():
                    if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                        cfg[k].update(v)
                    else:
                        cfg[k] = v
        except Exception as exc:
            print(f"[WARN] Could not parse {path}: {exc}. Using default config.")
    return cfg


def save_config(cfg: Dict[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def build_region_entry(
    px_rect: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    log_w: int,
    log_h: int,
) -> Dict[str, Any]:
    px_x, px_y, px_w, px_h = px_rect
    scale_x = img_w / float(log_w)
    scale_y = img_h / float(log_h)

    return {
        "pixel_rect": {"x": int(px_x), "y": int(px_y), "w": int(px_w), "h": int(px_h)},
        "logical_rect": {
            "x": round(px_x / scale_x, 2),
            "y": round(px_y / scale_y, 2),
            "w": round(px_w / scale_x, 2),
            "h": round(px_h / scale_y, 2),
        },
        "normalized_rect": {
            "nx": round(px_x / float(img_w), 6),
            "ny": round(px_y / float(img_h), 6),
            "nw": round(px_w / float(img_w), 6),
            "nh": round(px_h / float(img_h), 6),
        },
    }


class SmoothVisualSetupStudio:
    WINDOW_NAME = "TikTok DM Setup Studio (1/2/3: Region | R or Click Button: Refresh Screen | S: Save | Q: Quit)"

    def __init__(
        self,
        config: Dict[str, Any],
        config_path: Path,
        recipients: List[RecipientEntry],
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.recipients = recipients

        self.log_w, self.log_h = get_logical_display_size()
        self.screenshot_bgr: np.ndarray = capture_fullscreen_bgr()
        self.img_h, self.img_w = self.screenshot_bgr.shape[:2]

        self.scale_x = self.img_w / float(self.log_w)
        self.scale_y = self.img_h / float(self.log_h)

        max_total_w = min(1180, int(self.log_w * 0.88))
        self.right_pane_w = 410
        self.left_pane_w = max(560, max_total_w - self.right_pane_w)
        self.view_scale = self.left_pane_w / float(self.img_w)
        self.left_pane_h = int(round(self.img_h * self.view_scale))
        self.canvas_w = self.left_pane_w + self.right_pane_w
        self.canvas_h = max(self.left_pane_h + 34, 580)

        # Pre-scaled left screenshot cache (eliminates cv2.resize lag during drag!)
        self.scaled_screen_cache: np.ndarray = cv2.resize(
            self.screenshot_bgr,
            (self.left_pane_w, self.left_pane_h),
            interpolation=cv2.INTER_AREA,
        )
        self.static_canvas_cache: Optional[np.ndarray] = None
        self.needs_static_redraw: bool = True
        self.needs_display_present: bool = True

        # Clickable Refresh Button bounds in the right panel: (x1, y1, x2, y2)
        self.refresh_btn_rect: Tuple[int, int, int, int] = (
            self.left_pane_w + 12,
            44,
            self.canvas_w - 12,
            76,
        )

        self.active_region_key: str = "dm_sidebar"
        self.dragging: bool = False
        self.drag_start_px: Tuple[int, int] = (0, 0)
        self.drag_curr_px: Tuple[int, int] = (0, 0)
        self.status_message: str = (
            "Drag box on left screenshot. Wrong tab? Press [R] or click [REFRESH SCREEN] button!"
        )
        self.ocr_cache: Dict[str, List[OCRBox]] = {k: [] for k in REGION_KEYS}
        self._ocr_lock = threading.Lock()

        self._restore_saved_regions_if_compatible()
        self.trigger_async_ocr_all()

    def _restore_saved_regions_if_compatible(self) -> None:
        regions = self.config.get("regions", {})
        for key in REGION_KEYS:
            reg = regions.get(key)
            if not isinstance(reg, dict):
                continue
            norm = reg.get("normalized_rect")
            if isinstance(norm, dict):
                px_x = int(round(float(norm.get("nx", 0.0)) * self.img_w))
                px_y = int(round(float(norm.get("ny", 0.0)) * self.img_h))
                px_w = int(round(float(norm.get("nw", 0.0)) * self.img_w))
                px_h = int(round(float(norm.get("nh", 0.0)) * self.img_h))
                if px_w > 10 and px_h > 10:
                    self.config["regions"][key] = build_region_entry(
                        (px_x, px_y, px_w, px_h),
                        self.img_w,
                        self.img_h,
                        self.log_w,
                        self.log_h,
                    )

    def _view_to_image_px(self, vx: int, vy: int) -> Tuple[int, int]:
        clamped_vx = max(0, min(self.left_pane_w - 1, vx))
        clamped_vy = max(0, min(self.left_pane_h - 1, vy))
        px_x = int(round(clamped_vx / self.view_scale))
        px_y = int(round(clamped_vy / self.view_scale))
        return (
            max(0, min(self.img_w - 1, px_x)),
            max(0, min(self.img_h - 1, px_y)),
        )

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        # Check if user clicked the [REFRESH TIKTOK SCREEN (R)] button on the right panel
        bx1, by1, bx2, by2 = self.refresh_btn_rect
        if event == cv2.EVENT_LBUTTONDOWN and bx1 <= x <= bx2 and by1 <= y <= by2:
            self.recapture_live_screen()
            return

        if x >= self.left_pane_w or y >= self.left_pane_h:
            if event == cv2.EVENT_LBUTTONUP and self.dragging:
                self.dragging = False
                self.needs_display_present = True
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start_px = self._view_to_image_px(x, y)
            self.drag_curr_px = self.drag_start_px
            self.needs_display_present = True

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            new_pt = self._view_to_image_px(x, y)
            if new_pt != self.drag_curr_px:
                self.drag_curr_px = new_pt
                self.needs_display_present = True

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.drag_curr_px = self._view_to_image_px(x, y)
            x1, y1 = self.drag_start_px
            x2, y2 = self.drag_curr_px
            rx = min(x1, x2)
            ry = min(y1, y2)
            rw = abs(x2 - x1)
            rh = abs(y2 - y1)

            if rw < 14 or rh < 14:
                self.status_message = (
                    f"[Ignored] Selection for {self.active_region_key} was too small."
                )
                self.needs_static_redraw = True
                self.needs_display_present = True
                return

            key = self.active_region_key
            self.config["regions"][key] = build_region_entry(
                (rx, ry, rw, rh),
                self.img_w,
                self.img_h,
                self.log_w,
                self.log_h,
            )
            meta = REGION_META[key]
            self.status_message = (
                f"Updated {meta['short']} -> Logical: "
                f"{self.config['regions'][key]['logical_rect']} (running OCR...)"
            )
            self.needs_static_redraw = True
            self.needs_display_present = True
            self.trigger_async_ocr_for_key(key)

    def trigger_async_ocr_for_key(self, key: str) -> None:
        """Run Vision OCR in a background thread so mouse release is 100% lag-free."""
        reg = self.config.get("regions", {}).get(key)
        if not isinstance(reg, dict):
            return
        px = reg["pixel_rect"]
        crop_copy = self.screenshot_bgr[
            px["y"] : px["y"] + px["h"], px["x"] : px["x"] + px["w"]
        ].copy()

        def _worker() -> None:
            boxes = run_vision_ocr_on_bgr(crop_copy)
            with self._ocr_lock:
                self.ocr_cache[key] = boxes
                self.needs_static_redraw = True
                self.needs_display_present = True

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def trigger_async_ocr_all(self) -> None:
        for k in REGION_KEYS:
            self.trigger_async_ocr_for_key(k)

    def recapture_live_screen(self) -> None:
        """
        Hide the OpenCV window briefly so it doesn't photobomb the screenshot,
        switch Chrome to the TikTok Messages tab (or whatever tab you have active),
        capture a fresh screenshot, and bring the OpenCV window right back!
        """
        self.status_message = "Refreshing live screen from Google Chrome..."
        self.needs_static_redraw = True
        cv2.imshow(self.WINDOW_NAME, self.build_static_canvas())
        cv2.waitKey(15)

        # 1. Hide our Python GUI app so it doesn't cover Chrome
        app = AppKit.NSApplication.sharedApplication()
        app.hide_(None)
        time.sleep(0.15)

        # 2. Activate Chrome and auto-switch to tiktok.com/messages if open in any tab
        browser = self.config.get("browser_app", "Google Chrome")
        focus_chrome_and_switch_to_tiktok(browser, switch_to_tiktok_tab=True)
        time.sleep(0.45)

        # 3. Take fresh screenshot
        self.log_w, self.log_h = get_logical_display_size()
        self.screenshot_bgr = capture_fullscreen_bgr()
        self.img_h, self.img_w = self.screenshot_bgr.shape[:2]
        self.scale_x = self.img_w / float(self.log_w)
        self.scale_y = self.img_h / float(self.log_h)

        # 4. Update cached scaled image
        self.scaled_screen_cache = cv2.resize(
            self.screenshot_bgr,
            (self.left_pane_w, self.left_pane_h),
            interpolation=cv2.INTER_AREA,
        )

        # 5. Bring our Python OpenCV window right back to the front
        app.unhideWithoutActivation()
        app.activateIgnoringOtherApps_(True)

        self.status_message = (
            f"Refreshed live screen ({self.img_w}x{self.img_h} px) at {time.strftime('%H:%M:%S')}!"
        )
        self.needs_static_redraw = True
        self.needs_display_present = True
        self.trigger_async_ocr_all()

    def build_static_canvas(self) -> np.ndarray:
        """
        Build the composite canvas (left screenshot with saved boxes + right inspector).
        Cached and reused across frames so dragging a rubber-band box is 0.2ms fast.
        """
        canvas = np.full((self.canvas_h, self.canvas_w, 3), 24, dtype=np.uint8)
        scaled_screen = self.scaled_screen_cache.copy()

        with self._ocr_lock:
            ocr_snapshot = {k: list(v) for k, v in self.ocr_cache.items()}

        for key in REGION_KEYS:
            reg = self.config.get("regions", {}).get(key)
            meta = REGION_META[key]
            color = meta["color_bgr"]
            if isinstance(reg, dict):
                px = reg["pixel_rect"]
                vx1 = int(round(px["x"] * self.view_scale))
                vy1 = int(round(px["y"] * self.view_scale))
                vx2 = int(round((px["x"] + px["w"]) * self.view_scale))
                vy2 = int(round((px["y"] + px["h"]) * self.view_scale))
                thickness = 3 if key == self.active_region_key else 2
                cv2.rectangle(scaled_screen, (vx1, vy1), (vx2, vy2), color, thickness)

                tag_text = f"{meta['key_hint']}: {meta['short']}"
                cv2.rectangle(
                    scaled_screen,
                    (vx1, max(0, vy1 - 20)),
                    (min(self.left_pane_w - 1, vx1 + 185), vy1),
                    color,
                    -1,
                )
                cv2.putText(
                    scaled_screen,
                    tag_text,
                    (vx1 + 4, max(13, vy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (15, 15, 15),
                    1,
                    cv2.LINE_AA,
                )

                for box in ocr_snapshot.get(key, []):
                    bx1 = int(round((px["x"] + box.px_x) * self.view_scale))
                    by1 = int(round((px["y"] + box.px_y) * self.view_scale))
                    bx2 = int(round((px["x"] + box.px_x + box.px_w) * self.view_scale))
                    by2 = int(round((px["y"] + box.px_y + box.px_h) * self.view_scale))
                    cv2.rectangle(
                        scaled_screen, (bx1, by1), (bx2, by2), (255, 80, 255), 1
                    )

        canvas[0 : self.left_pane_h, 0 : self.left_pane_w] = scaled_screen

        status_y = min(self.canvas_h - 10, self.left_pane_h + 21)
        cv2.putText(
            canvas,
            self.status_message[:118],
            (10, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (220, 235, 250),
            1,
            cv2.LINE_AA,
        )

        # Right Inspector Panel
        rx0 = self.left_pane_w + 12
        pw = self.right_pane_w - 24

        cv2.putText(
            canvas,
            f"TIKTOK DM STUDIO ({len(self.recipients)} in nicknames.txt)",
            (rx0, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "[1/2/3] Select Region   [S] Save   [Q] Quit",
            (rx0, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (175, 215, 255),
            1,
            cv2.LINE_AA,
        )

        # Draw Clickable [REFRESH TIKTOK SCREEN (Press R or Click)] Button
        bx1, by1, bx2, by2 = self.refresh_btn_rect
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (55, 165, 75), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (120, 240, 140), 1)
        cv2.putText(
            canvas,
            "REFRESH / SWITCH TO TIKTOK [Press R or Click]",
            (bx1 + 12, by1 + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y_cursor = by2 + 10
        card_h = max(140, int((self.canvas_h - y_cursor - 28) / 3))

        for key in REGION_KEYS:
            meta = REGION_META[key]
            color = meta["color_bgr"]
            is_active = key == self.active_region_key

            cv2.rectangle(
                canvas,
                (rx0, y_cursor),
                (rx0 + pw, y_cursor + card_h),
                color if is_active else (75, 75, 75),
                2 if is_active else 1,
            )

            prefix = ">> " if is_active else "   "
            cv2.putText(
                canvas,
                f"{prefix}{meta['label']}",
                (rx0 + 6, y_cursor + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.41,
                color,
                1,
                cv2.LINE_AA,
            )

            reg = self.config.get("regions", {}).get(key)
            if not isinstance(reg, dict):
                cv2.putText(
                    canvas,
                    f"Not set. Press [{meta['key_hint']}] & drag on screenshot.",
                    (rx0 + 10, y_cursor + int(card_h / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.39,
                    (150, 150, 150),
                    1,
                    cv2.LINE_AA,
                )
            else:
                lrect = reg["logical_rect"]
                prect = reg["pixel_rect"]
                cv2.putText(
                    canvas,
                    f"Logical pt: ({lrect['x']}, {lrect['y']}, {lrect['w']}x{lrect['h']})",
                    (rx0 + 6, y_cursor + 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (205, 205, 205),
                    1,
                    cv2.LINE_AA,
                )

                crop = self.screenshot_bgr[
                    prect["y"] : prect["y"] + prect["h"],
                    prect["x"] : prect["x"] + prect["w"],
                ]
                thumb_max_w = pw - 14
                thumb_max_h = max(42, card_h - 58)
                if crop.size > 0:
                    ch, cw = crop.shape[:2]
                    scale = min(thumb_max_w / float(cw), thumb_max_h / float(ch))
                    tw = max(1, int(round(cw * scale)))
                    th = max(1, int(round(ch * scale)))
                    thumb = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
                    ty0 = y_cursor + 38
                    tx0 = rx0 + 6
                    canvas[ty0 : ty0 + th, tx0 : tx0 + tw] = thumb
                    cv2.rectangle(
                        canvas, (tx0, ty0), (tx0 + tw, ty0 + th), (95, 95, 95), 1
                    )

                ocr_boxes = ocr_snapshot.get(key, [])
                ocr_summary = (
                    " | ".join(b.text for b in ocr_boxes[:4])
                    if ocr_boxes
                    else "(No OCR text)"
                )
                cv2.putText(
                    canvas,
                    f"OCR ({len(ocr_boxes)}): {ocr_summary[:52]}",
                    (rx0 + 6, y_cursor + card_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (180, 245, 190) if ocr_boxes else (130, 140, 150),
                    1,
                    cv2.LINE_AA,
                )

            y_cursor += card_h + 7

        self.static_canvas_cache = canvas
        self.needs_static_redraw = False
        return canvas

    def get_frame_for_display(self) -> np.ndarray:
        """
        Return the current frame to display. If dragging, overlays the live
        selection rectangle onto a fast copy of `static_canvas_cache`.
        """
        if self.needs_static_redraw or self.static_canvas_cache is None:
            self.build_static_canvas()

        if not self.dragging:
            return self.static_canvas_cache  # type: ignore[return-value]

        frame = self.static_canvas_cache.copy()  # type: ignore[union-attr]
        color = REGION_META[self.active_region_key]["color_bgr"]
        dx1 = int(
            round(min(self.drag_start_px[0], self.drag_curr_px[0]) * self.view_scale)
        )
        dy1 = int(
            round(min(self.drag_start_px[1], self.drag_curr_px[1]) * self.view_scale)
        )
        dx2 = int(
            round(max(self.drag_start_px[0], self.drag_curr_px[0]) * self.view_scale)
        )
        dy2 = int(
            round(max(self.drag_start_px[1], self.drag_curr_px[1]) * self.view_scale)
        )
        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 2)

        # Show live logical dimensions next to cursor while dragging
        lw = int(round(abs(self.drag_curr_px[0] - self.drag_start_px[0]) / self.scale_x))
        lh = int(round(abs(self.drag_curr_px[1] - self.drag_start_px[1]) / self.scale_y))
        cv2.putText(
            frame,
            f"{lw}x{lh} pt",
            (min(self.left_pane_w - 85, dx2 + 6), max(16, dy2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        return frame

    def save_current_calibration(self) -> bool:
        missing = [
            k
            for k in REGION_KEYS
            if not isinstance(self.config["regions"].get(k), dict)
        ]
        if missing:
            self.status_message = (
                f"[Cannot Save] Missing region(s): {', '.join(missing)}"
            )
            self.needs_static_redraw = True
            self.needs_display_present = True
            return False

        self.config["display_profile"] = {
            "logical_width": self.log_w,
            "logical_height": self.log_h,
            "image_width": self.img_w,
            "image_height": self.img_h,
            "scale_x": round(self.scale_x, 4),
            "scale_y": round(self.scale_y, 4),
            "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        save_config(self.config, self.config_path)
        self.status_message = f"Saved calibration to {self.config_path.resolve()}!"
        self.needs_static_redraw = True
        self.needs_display_present = True
        print(f"[OK] Saved calibration to {self.config_path.resolve()}")
        return True

    def run_loop(self) -> bool:
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW_NAME, self.on_mouse)

        saved = False
        try:
            while True:
                if self.needs_display_present or self.needs_static_redraw:
                    frame = self.get_frame_for_display()
                    cv2.imshow(self.WINDOW_NAME, frame)
                    self.needs_display_present = False

                # 5ms poll -> ~120Hz mouse responsiveness for smooth box dragging
                key = cv2.waitKey(5) & 0xFF
                if key == 255:
                    continue

                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key == ord("1"):
                    self.active_region_key = "dm_sidebar"
                    self.status_message = (
                        "Active: 1. DM Sidebar — Drag box over conversation list."
                    )
                    self.needs_static_redraw = True
                    self.needs_display_present = True
                elif key == ord("2"):
                    self.active_region_key = "recipient_header"
                    self.status_message = (
                        "Active: 2. Recipient Header — Cover Display Name AND @username!"
                    )
                    self.needs_static_redraw = True
                    self.needs_display_present = True
                elif key == ord("3"):
                    self.active_region_key = "message_input"
                    self.status_message = (
                        "Active: 3. Message Input Box — Drag over 'Send a message...'."
                    )
                    self.needs_static_redraw = True
                    self.needs_display_present = True
                elif key in (ord("r"), ord("R"), ord(" ")):
                    self.recapture_live_screen()
                elif key in (ord("v"), ord("V")):
                    self.trigger_async_ocr_all()
                elif key in (ord("s"), ord("S")):
                    if self.save_current_calibration():
                        saved = True
        finally:
            cv2.destroyAllWindows()
            for _ in range(4):
                cv2.waitKey(1)

        return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smooth 120Hz TikTok DM Region Calibrator with Live Tab Refresh."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT_PATH)
    parser.add_argument("--nicknames", type=Path, default=NICKNAMES_DEFAULT_PATH)
    parser.add_argument("--set-message", type=str, default=None)
    parser.add_argument("--no-focus", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_or_init_config(args.config)
    cfg["nicknames_file"] = str(args.nicknames)

    recipients = ensure_and_parse_nicknames(args.nicknames)
    print(f"[INFO] Loaded {len(recipients)} recipients from '{args.nicknames}'.")

    if args.set_message is not None:
        cfg["message"] = args.set_message
        save_config(cfg, args.config)
        print(f"[OK] Updated message to: {cfg['message']}")

    if not args.no_focus:
        browser = cfg.get("browser_app", "Google Chrome")
        print(f"[INFO] Focusing '{browser}' and switching to TikTok Messages tab...")
        focus_chrome_and_switch_to_tiktok(browser, switch_to_tiktok_tab=True)
        time.sleep(0.6)

    studio = SmoothVisualSetupStudio(cfg, args.config, recipients)
    studio.run_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())