#!/usr/bin/env python3
"""
automation.py — macOS TikTok DM Automation Engine
Fixes:
  1. Emoji-only Blank Rows (e.g. blank 8-space name with only '🔥' and '23:23'):
     Correctly measures white pixels at the timestamp's Y-level when no preview
     text exists, so blank-name rows with emoji-only previews are never skipped.
  2. Tiny Symbol 'π' (@paingpaing1123):
     Handles both OCR text aliases ('TT', 'IT', 'n', 'r', 'm', 'pi') AND cases
     where Vision OCR completely misses the tiny 'π' glyph above 'sent a sticker'
     by detecting compact symbol pixel clusters (7..115 white px) in the name slot.
  3. Short 1-2 char names ('M', 'K', 'w', 'F.') & clipped 'zarı' (dotless 'ı' -> 'i').
  4. Zero random clicks + Instant Stop on ESC / F12 / Physical Mouse Movement.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import AppKit
import cv2
import numpy as np
import Quartz
import Vision
from Foundation import NSURL
from pynput import keyboard as pynput_keyboard
from pynput import mouse as pynput_mouse

CONFIG_DEFAULT_PATH = Path("config.json")
NICKNAMES_DEFAULT_PATH = Path("nicknames.txt")
HISTORY_DEFAULT_PATH = Path("sent_history.json")
LOG_DEFAULT_PATH = Path("automation.log")

TIKTOK_MESSAGES_URL = "https://www.tiktok.com/messages"

# macOS Virtual Keycodes
KVK_ANSI_V = 0x09   # 9
KVK_RETURN = 0x24   # 36
KVK_COMMAND = 0x37  # 55

# Vision OCR aliases for Greek pi (π)
PI_OCR_ALIASES = {
    "π", "∏", "pi", "p1", "tt", "t t", "ii", "11",
    "it", "ti", "tl", "1t", "77", "n", "r", "m", "u", "ll"
}

# Clock timestamp (means TODAY in TikTok DM sidebar, e.g. '23:23', '23:07')
CLOCK_TIME_REGEX = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")

# Relative today timestamps ('Just now', '5m', '2h')
RELATIVE_TODAY_REGEX = re.compile(
    r"\b(just now|\d+\s*[smh])\b", re.IGNORECASE
)

# Calendar date timestamp in TikTok DM sidebar ('10/26/2026', '9/27/2026', '2026-10-26')
SLASH_DATE_REGEX = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
ISO_DATE_REGEX = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")

# Message preview / reaction / streak status patterns (Line 2 in TikTok sidebar)
PREVIEW_STATUS_REGEX = re.compile(
    r"(^you:|\bsent a\b|\bshared a\b|\bshared photos\b|\breacted\b|\breplied to\b|"
    r"\bstreak\b|\bthis message\b|\bcamera mem|\bsay hi\b|\bwave\b|\bseen\b|"
    r"\bactive\b|\btyping\b|\bnew message\b|\bsticker\b|\bto your\b|\bshut up\b|"
    r"\binstock\b|\bhi commision\b|\bartwork\b)",
    re.IGNORECASE,
)


@dataclass
class OCRLine:
    text: str
    confidence: float
    px_x: int
    px_y: int
    px_w: int
    px_h: int
    p95_brightness: float = 255.0

    @property
    def center_px(self) -> Tuple[float, float]:
        return (self.px_x + self.px_w / 2.0, self.px_y + self.px_h / 2.0)


@dataclass
class RecipientSpec:
    display_name: str
    username: str
    clean_display: str
    placeholder_name: str

    @property
    def permanent_id(self) -> str:
        return self.username.lower()

    @property
    def handle_bare(self) -> str:
        return self.username.lower().lstrip("@")

    @property
    def is_invisible(self) -> bool:
        return len(self.clean_display) == 0

    @property
    def is_unique_symbol(self) -> bool:
        return self.display_name.strip() in ("π", ".")

    @property
    def debug_category(self) -> str:
        if self.is_invisible:
            return "Invisible 8-Spaces (Blank Name / Orphan Preview)"
        if self.display_name.strip() == "π":
            return "Symbol (π / OCR 'TT')"
        if self.display_name.strip() == ".":
            return "Single Dot (.)"
        if len(self.clean_display) <= 4:
            return f"Short Display Name ('{self.clean_display}')"
        return f"Full Display Name ('{self.clean_display}')"


@dataclass
class ConversationRow:
    top_y_log: float
    click_y_log: float
    click_x_log: float
    lines: List[OCRLine] = field(default_factory=list)
    name_candidates: List[str] = field(default_factory=list)
    preview_texts: List[str] = field(default_factory=list)
    row_date: Optional[date] = None
    is_today_or_recent: bool = True
    is_blank_or_orphan_preview: bool = False
    is_dot_candidate: bool = False
    is_pi_or_symbol_candidate: bool = False
    white_px_in_name_zone: int = -1

    def raw_summary(self) -> str:
        raw = " / ".join(l.text for l in self.lines)
        date_tag = self.row_date.isoformat() if self.row_date else "no-date"
        flags = []
        if self.is_blank_or_orphan_preview:
            flags.append(f"BLANK(wpx={self.white_px_in_name_zone})")
        if self.is_dot_candidate:
            flags.append(f"DOT(wpx={self.white_px_in_name_zone})")
        if self.is_pi_or_symbol_candidate:
            flags.append(f"SYM(wpx={self.white_px_in_name_zone})")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        return f"[{date_tag}]{flag_str} {raw}"


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


def normalize_ocr_confusions(text: str) -> str:
    """
    Normalize common Vision OCR character confusions on short display names
    (e.g. 'zarı' -> 'zari', 'chl.' -> 'chi.', 'kise' -> 'klse').
    """
    s = text.lower().strip()
    s = re.sub(r"^[•·.\-_|:\s]+|[•·.\-_|:\s]+$", "", s)
    # Include Turkish dotless 'ı' (U+0131) when top of 'i' is clipped near "Messages" header
    s = (
        s.replace("ı", "i")
        .replace("1", "i")
        .replace("l", "i")
        .replace("|", "i")
        .replace("!", "i")
    )
    s = s.replace("0", "o").replace("5", "s").replace("2", "z")
    return s


def strip_timestamps_and_bullets(text: str) -> str:
    """
    Remove inline timestamps ('• 21:44', '· 10/26/2026', '23:23') from an OCR line
    WITHOUT discarding the Display Name that precedes it.
    """
    s = text.strip()
    if "•" in s:
        left = s.split("•", 1)[0].strip()
        if left:
            s = left
    if "·" in s:
        left = s.split("·", 1)[0].strip()
        if left:
            s = left
    s = SLASH_DATE_REGEX.sub("", s)
    s = ISO_DATE_REGEX.sub("", s)
    s = CLOCK_TIME_REGEX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_date_from_text(text: str, today: date) -> Optional[date]:
    s = text.strip()
    m_slash = SLASH_DATE_REGEX.search(s)
    if m_slash:
        mm, dd, yy = int(m_slash.group(1)), int(m_slash.group(2)), int(m_slash.group(3))
        if yy < 100:
            yy += 2000
        try:
            return date(yy, mm, dd)
        except ValueError:
            pass

    m_iso = ISO_DATE_REGEX.search(s)
    if m_iso:
        yy, mm, dd = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        try:
            return date(yy, mm, dd)
        except ValueError:
            pass

    if CLOCK_TIME_REGEX.search(s) or RELATIVE_TODAY_REGEX.search(s):
        return today

    if "yesterday" in s.lower():
        return date.fromordinal(today.toordinal() - 1)

    return None


def parse_nicknames_file(path: Path) -> List[RecipientSpec]:
    if not path.exists():
        raise FileNotFoundError(
            f"Recipients file '{path}' not found. Run 'python3 setup.py' first."
        )
    specs: List[RecipientSpec] = []
    seen: Set[str] = set()

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
        if perm_id in seen:
            continue
        seen.add(perm_id)

        clean_disp = strip_invisible_and_emojis(disp)
        placeholder = clean_disp if clean_disp else f"<invisible:{handle}>"

        specs.append(
            RecipientSpec(
                display_name=disp,
                username=handle,
                clean_display=clean_disp,
                placeholder_name=placeholder,
            )
        )
    return specs


# --- Chrome & TikTok Messages Tab Controller ---


def ensure_chrome_and_tiktok_messages_tab(
    logger: logging.Logger,
    browser_app: str = "Google Chrome",
    target_url: str = TIKTOK_MESSAGES_URL,
) -> bool:
    applescript = f"""
    tell application "{browser_app}"
        activate
        if (count of windows) is 0 then
            make new window
            set URL of active tab of front window to "{target_url}"
            return "OPENED_NEW_WINDOW"
        end if

        set activeURL to URL of active tab of front window
        if activeURL contains "tiktok.com/messages" then
            return "ALREADY_ACTIVE"
        end if

        repeat with w in windows
            set tabIdx to 1
            repeat with t in tabs of w
                if (URL of t) contains "tiktok.com/messages" then
                    set active tab index of w to tabIdx
                    set index of w to 1
                    return "SWITCHED_EXISTING_TAB"
                end if
                set tabIdx to tabIdx + 1
            end repeat
        end repeat

        set URL of active tab of front window to "{target_url}"
        return "NAVIGATED_ACTIVE_TAB"
    end tell
    """
    try:
        res = subprocess.run(
            ["/usr/bin/osascript", "-e", applescript],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        action = res.stdout.strip()
        logger.info(f"Browser Tab Check -> {action}")

        if action in ("OPENED_NEW_WINDOW", "NAVIGATED_ACTIVE_TAB"):
            logger.info("Waiting for TikTok Messages page to finish loading...")
            wait_script = f"""
            tell application "{browser_app}"
                repeat 35 times
                    if loading of active tab of front window is false then
                        exit repeat
                    end if
                    delay 0.2
                end repeat
            end tell
            """
            subprocess.run(
                ["/usr/bin/osascript", "-e", wait_script],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(2.2)
        elif action == "SWITCHED_EXISTING_TAB":
            time.sleep(0.45)
        else:
            time.sleep(0.20)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(f"Failed to control {browser_app}: {exc.stderr}")
        return False


class SafetyController:
    """
    Instant Emergency Stop on:
      1. Pressing ESC or F12
      2. Any physical mouse movement by the user (> 12px away from programmatic cursor)
      3. Moving mouse to top-left corner (0, 0)
    Plus Pause/Resume on F8.
    """

    def __init__(self, logger: logging.Logger, mouse_move_tolerance_px: float = 12.0) -> None:
        self.logger = logger
        self.mouse_move_tolerance_px = mouse_move_tolerance_px
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.stop_reason: str = ""
        self._kb_listener: Optional[pynput_keyboard.Listener] = None
        self._mouse_listener: Optional[pynput_mouse.Listener] = None
        self._lock = threading.Lock()
        self._expected_mouse_pos: Optional[Tuple[float, float]] = None
        self._programmatic_move_until: float = 0.0

    def start(self) -> None:
        self._expected_mouse_pos = get_current_mouse_logical_pos()
        self._kb_listener = pynput_keyboard.Listener(on_press=self._on_key_press)
        self._kb_listener.daemon = True
        self._kb_listener.start()

        self._mouse_listener = pynput_mouse.Listener(on_move=self._on_mouse_move)
        self._mouse_listener.daemon = True
        self._mouse_listener.start()

        self.logger.info(
            "Safety Active -> [ESC / F12 / Physical Mouse Move]: Instant Stop | [F8]: Pause/Resume"
        )

    def stop(self) -> None:
        if self._kb_listener is not None:
            self._kb_listener.stop()
        if self._mouse_listener is not None:
            self._mouse_listener.stop()

    def begin_programmatic_mouse_move(self, target_x: float, target_y: float, duration: float = 0.08) -> None:
        with self._lock:
            self._expected_mouse_pos = (target_x, target_y)
            self._programmatic_move_until = time.monotonic() + duration

    def _trigger_stop(self, reason: str) -> None:
        if not self.stop_event.is_set():
            self.stop_reason = reason
            self.stop_event.set()
            self.pause_event.clear()
            self.logger.critical(f"[EMERGENCY STOP] {reason}")

    def _on_key_press(self, key: Any) -> None:
        try:
            if key in (pynput_keyboard.Key.esc, pynput_keyboard.Key.f12):
                key_name = "ESC" if key == pynput_keyboard.Key.esc else "F12"
                self._trigger_stop(f"{key_name} key pressed by user.")
            elif key == pynput_keyboard.Key.f8:
                if self.pause_event.is_set():
                    with self._lock:
                        self._expected_mouse_pos = get_current_mouse_logical_pos()
                    self.pause_event.clear()
                    self.logger.info("[HOTKEY F8] RESUMED.")
                else:
                    self.pause_event.set()
                    self.logger.warning("[HOTKEY F8] PAUSED. Press F8 to resume.")
        except Exception:
            pass

    def _on_mouse_move(self, x: float, y: float) -> None:
        if self.stop_event.is_set() or self.pause_event.is_set():
            return
        if x <= 2.0 and y <= 2.0:
            self._trigger_stop("Mouse moved to top-left failsafe corner (0, 0).")
            return

        now = time.monotonic()
        with self._lock:
            if now <= self._programmatic_move_until:
                self._expected_mouse_pos = (float(x), float(y))
                return
            exp = self._expected_mouse_pos

        if exp is not None:
            dist = math.hypot(float(x) - exp[0], float(y) - exp[1])
            if dist > self.mouse_move_tolerance_px:
                self._trigger_stop(
                    f"User moved mouse manually (detected at ({x:.0f}, {y:.0f}), "
                    f"expected ({exp[0]:.0f}, {exp[1]:.0f}), delta={dist:.1f}px)."
                )

    def check_or_raise(self) -> None:
        if self.stop_event.is_set():
            raise InterruptedError(self.stop_reason or "Emergency stop active.")

        mx, my = get_current_mouse_logical_pos()
        if mx <= 2.0 and my <= 2.0:
            self._trigger_stop("Mouse corner failsafe triggered at (0, 0).")
            raise InterruptedError(self.stop_reason)

        with self._lock:
            exp = self._expected_mouse_pos
            in_prog_window = time.monotonic() <= self._programmatic_move_until

        if not in_prog_window and exp is not None and not self.pause_event.is_set():
            dist = math.hypot(mx - exp[0], my - exp[1])
            if dist > self.mouse_move_tolerance_px:
                self._trigger_stop(
                    f"User moved mouse manually (at ({mx:.0f}, {my:.0f}), "
                    f"expected ({exp[0]:.0f}, {exp[1]:.0f}), delta={dist:.1f}px)."
                )
                raise InterruptedError(self.stop_reason)

        while self.pause_event.is_set():
            if self.stop_event.is_set():
                raise InterruptedError(self.stop_reason or "Emergency stop triggered while paused.")
            time.sleep(0.05)

    def sleep_fast(self, duration: float) -> None:
        end_time = time.monotonic() + max(0.0, duration)
        while time.monotonic() < end_time:
            self.check_or_raise()
            remaining = end_time - time.monotonic()
            time.sleep(min(0.025, max(0.004, remaining)))


def setup_logger(log_path: Path, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("tiktok_dm_fixed")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def get_current_mouse_logical_pos() -> Tuple[float, float]:
    event = Quartz.CGEventCreate(None)
    loc = Quartz.CGEventGetLocation(event)
    return float(loc.x), float(loc.y)


def is_browser_frontmost_fast(target_app: str = "Google Chrome") -> bool:
    ws = AppKit.NSWorkspace.sharedWorkspace()
    front_app = ws.frontmostApplication()
    if front_app is None:
        return True
    name = str(front_app.localizedName() or "")
    return target_app.lower() in name.lower()


def capture_region_fast_bgr(
    log_x: float, log_y: float, log_w: float, log_h: float
) -> np.ndarray:
    rx = max(0, int(round(log_x)))
    ry = max(0, int(round(log_y)))
    rw = max(16, int(round(log_w)))
    rh = max(16, int(round(log_h)))
    rect_arg = f"{rx},{ry},{rw},{rh}"

    fd, tmp_path = tempfile.mkstemp(prefix="tiktok_reg_", suffix=".png")
    os.close(fd)
    try:
        res = subprocess.run(
            ["/usr/sbin/screencapture", "-x", "-R", rect_arg, tmp_path],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if res.returncode != 0 or not os.path.exists(tmp_path):
            raise RuntimeError("screencapture -R failed.")
        img = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise RuntimeError("Captured region image is empty.")
        return img
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def run_vision_ocr_with_brightness(crop_bgr: np.ndarray) -> List[OCRLine]:
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    img_h, img_w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    fd, tmp_path = tempfile.mkstemp(prefix="tiktok_ocr_", suffix=".png")
    os.close(fd)

    lines: List[OCRLine] = []
    try:
        cv2.imwrite(tmp_path, crop_bgr)
        url = NSURL.fileURLWithPath_(tmp_path)
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)
        if hasattr(request, "setMinimumTextHeight_"):
            request.setMinimumTextHeight_(0.0)

        ok, _err = handler.performRequests_error_([request], None)
        if not ok or request.results() is None:
            return []

        for obs in request.results():
            cands = obs.topCandidates_(1)
            if not cands or len(cands) == 0:
                continue
            cand = cands[0]
            txt = str(cand.string()).strip()
            if not txt:
                continue
            conf = float(cand.confidence())
            bbox = obs.boundingBox()
            nx = float(bbox.origin.x)
            ny = float(bbox.origin.y)
            nw = float(bbox.size.width)
            nh = float(bbox.size.height)

            px_x = max(0, int(round(nx * img_w)))
            px_y = max(0, int(round((1.0 - ny - nh) * img_h)))
            px_w = max(1, int(round(nw * img_w)))
            px_h = max(1, int(round(nh * img_h)))

            sub_gray = gray[
                px_y : min(img_h, px_y + px_h), px_x : min(img_w, px_x + px_w)
            ]
            p95 = float(np.percentile(sub_gray, 95)) if sub_gray.size > 0 else 255.0

            lines.append(
                OCRLine(
                    text=txt,
                    confidence=conf,
                    px_x=px_x,
                    px_y=px_y,
                    px_w=px_w,
                    px_h=px_h,
                    p95_brightness=p95,
                )
            )
        lines.sort(key=lambda item: (item.px_y, item.px_x))
        return lines
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# --- Quartz Hardware Mouse & Keyboard Controllers ---


def quartz_fast_click(x: float, y: float, safety: SafetyController) -> None:
    safety.check_or_raise()
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    sx, sy = get_current_mouse_logical_pos()

    safety.begin_programmatic_mouse_move(x, y, duration=0.08)

    for frac in (0.5, 1.0):
        cx = sx + (x - sx) * frac
        cy = sy + (y - sy) * frac
        safety.begin_programmatic_mouse_move(cx, cy, duration=0.06)
        mv = Quartz.CGEventCreateMouseEvent(
            src, Quartz.kCGEventMouseMoved, (cx, cy), Quartz.kCGMouseButtonLeft
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, mv)
        time.sleep(0.006)

    safety.begin_programmatic_mouse_move(x, y, duration=0.08)
    down_ev = Quartz.CGEventCreateMouseEvent(
        src, Quartz.kCGEventLeftMouseDown, (x, y), Quartz.kCGMouseButtonLeft
    )
    Quartz.CGEventSetIntegerValueField(down_ev, Quartz.kCGMouseEventClickState, 1)

    up_ev = Quartz.CGEventCreateMouseEvent(
        src, Quartz.kCGEventLeftMouseUp, (x, y), Quartz.kCGMouseButtonLeft
    )
    Quartz.CGEventSetIntegerValueField(up_ev, Quartz.kCGMouseEventClickState, 1)

    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down_ev)
    time.sleep(0.022)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up_ev)
    safety.begin_programmatic_mouse_move(x, y, duration=0.05)


def quartz_scroll_sidebar_to_top(
    x: float, y: float, safety: SafetyController
) -> None:
    safety.check_or_raise()
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    safety.begin_programmatic_mouse_move(x, y, duration=0.18)
    mv = Quartz.CGEventCreateMouseEvent(
        src, Quartz.kCGEventMouseMoved, (x, y), Quartz.kCGMouseButtonLeft
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, mv)
    time.sleep(0.02)

    for _ in range(5):
        safety.check_or_raise()
        scroll_ev = Quartz.CGEventCreateScrollWheelEvent(
            src, Quartz.kCGScrollEventUnitPixel, 1, 350
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, scroll_ev)
        time.sleep(0.018)


def quartz_scroll_segment_down(
    x: float,
    y: float,
    segment_height_logical: float,
    safety: SafetyController,
) -> None:
    safety.check_or_raise()
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    safety.begin_programmatic_mouse_move(x, y, duration=0.18)
    mv = Quartz.CGEventCreateMouseEvent(
        src, Quartz.kCGEventMouseMoved, (x, y), Quartz.kCGMouseButtonLeft
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, mv)
    time.sleep(0.015)

    total_px = int(round(segment_height_logical * 0.62))
    step_px = max(35, total_px // 4)

    for _ in range(4):
        safety.check_or_raise()
        scroll_ev = Quartz.CGEventCreateScrollWheelEvent(
            src, Quartz.kCGScrollEventUnitPixel, 1, -step_px
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, scroll_ev)
        time.sleep(0.022)


def get_clipboard_text() -> Optional[str]:
    try:
        res = subprocess.run(
            ["/usr/bin/pbpaste"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return res.stdout.decode("utf-8", errors="replace")
    except Exception:
        return None


def set_clipboard_text(text: str) -> None:
    proc = subprocess.Popen(
        ["/usr/bin/pbcopy"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proc.communicate(input=text.encode("utf-8"))


def quartz_cmd_v_paste(safety: SafetyController) -> None:
    safety.check_or_raise()
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)

    cmd_down = Quartz.CGEventCreateKeyboardEvent(src, KVK_COMMAND, True)
    Quartz.CGEventSetFlags(cmd_down, Quartz.kCGEventFlagMaskCommand)

    v_down = Quartz.CGEventCreateKeyboardEvent(src, KVK_ANSI_V, True)
    Quartz.CGEventSetFlags(v_down, Quartz.kCGEventFlagMaskCommand)

    v_up = Quartz.CGEventCreateKeyboardEvent(src, KVK_ANSI_V, False)
    Quartz.CGEventSetFlags(v_up, Quartz.kCGEventFlagMaskCommand)

    cmd_up = Quartz.CGEventCreateKeyboardEvent(src, KVK_COMMAND, False)
    Quartz.CGEventSetFlags(cmd_up, 0)

    Quartz.CGEventPost(Quartz.kCGHIDEventTap, cmd_down)
    time.sleep(0.015)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, v_down)
    time.sleep(0.020)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, v_up)
    time.sleep(0.015)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, cmd_up)


def quartz_press_return(safety: SafetyController) -> None:
    safety.check_or_raise()
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    ret_down = Quartz.CGEventCreateKeyboardEvent(src, KVK_RETURN, True)
    Quartz.CGEventSetFlags(ret_down, 0)
    ret_up = Quartz.CGEventCreateKeyboardEvent(src, KVK_RETURN, False)
    Quartz.CGEventSetFlags(ret_up, 0)

    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ret_down)
    time.sleep(0.025)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ret_up)


# --- Stage 1: Sidebar Row Reconstruction & Strict Blank/Symbol Detection ---


def count_bright_white_pixels_in_name_slot(
    sb_gray: np.ndarray,
    x_name_px: int,
    y_center_px: int,
    scale_x: float,
    scale_y: float,
) -> int:
    img_h, img_w = sb_gray.shape[:2]
    half_h = max(7, int(round(11.0 * scale_y)))
    box_w = max(45, int(round(120.0 * scale_x)))

    y1 = max(0, y_center_px - half_h)
    y2 = min(img_h, y_center_px + half_h)
    x1 = max(0, x_name_px - int(round(6.0 * scale_x)))
    x2 = min(img_w, x1 + box_w)

    if y2 <= y1 or x2 <= x1:
        return 0
    patch = sb_gray[y1:y2, x1:x2]
    return int(np.sum(patch > 185))


def is_pure_timestamp_or_status_fragment(text: str) -> bool:
    s = text.strip()
    if not s:
        return True
    if re.fullmatch(r"[\d:/.\s•·\-apAPmM]+", s):
        return True
    if PREVIEW_STATUS_REGEX.search(s):
        return True
    return False


def analyze_sidebar_segment_rows(
    sb_img: np.ndarray,
    sb_lines: List[OCRLine],
    sb_log_x: float,
    sb_log_y: float,
    sb_log_w: float,
    scale_x: float,
    scale_y: float,
    today: date,
    max_chat_age_days: int = 1,
) -> List[ConversationRow]:
    """
    Reconstruct conversation rows from Vision OCR lines and accurately classify:
      1. Normal Display Name rows (including clipped 'zarı' and 1-char 'M').
      2. Emoji-only Blank Rows (e.g. only '23:23' on the right and '🔥' on the left).
      3. Tiny Symbol Rows ('π' or '.') even when Vision OCR only reads 'sent a sticker'
         and '23:07' and misses the small 'π' glyph above it.
    """
    sb_gray = cv2.cvtColor(sb_img, cv2.COLOR_BGR2GRAY)
    sb_px_h, sb_px_w = sb_gray.shape[:2]
    click_x_log = sb_log_x + sb_log_w * 0.42

    left_lines = [l.px_x for l in sb_lines if l.px_x < sb_px_w * 0.55]
    x_name_px = (
        int(np.percentile(left_lines, 20)) if left_lines else int(sb_px_w * 0.24)
    )

    raw_rows: List[ConversationRow] = []
    for line in sb_lines:
        _, cy_px = line.center_px
        y_log = sb_log_y + (cy_px / scale_y)

        if raw_rows and (y_log - raw_rows[-1].top_y_log) <= 32.0:
            raw_rows[-1].lines.append(line)
            raw_rows[-1].click_y_log = raw_rows[-1].top_y_log + 8.0
        else:
            raw_rows.append(
                ConversationRow(
                    top_y_log=y_log,
                    click_y_log=y_log,
                    click_x_log=click_x_log,
                    lines=[line],
                )
            )

    for row in raw_rows:
        for line in row.lines:
            dt = extract_date_from_text(line.text, today)
            if dt is not None:
                row.row_date = dt
                break

        if row.row_date is not None:
            age_days = (today - row.row_date).days
            row.is_today_or_recent = age_days <= max_chat_age_days
        else:
            row.is_today_or_recent = True

        top_y_px = row.lines[0].px_y
        top_tier_fragments: List[str] = []
        has_left_side_text = False

        for line in row.lines:
            raw_t = line.text.strip()
            if not raw_t:
                continue

            # Ignore right-hand timestamp tokens ('23:23', '23:07') for name candidates
            if line.px_x > sb_px_w * 0.60 and re.fullmatch(
                r"[\d:/.\s•·\-apAPmM]+", raw_t
            ):
                continue

            has_left_side_text = True

            # Explicit single dot '.' in the left name column
            if raw_t in (".", "·", "•") and line.px_x <= sb_px_w * 0.45:
                row.is_dot_candidate = True
                row.name_candidates.append(".")
                continue

            stripped_t = strip_timestamps_and_bullets(raw_t)
            if not stripped_t:
                continue

            if is_pure_timestamp_or_status_fragment(stripped_t) and line.p95_brightness < 215.0:
                row.preview_texts.append(stripped_t)
            else:
                row.name_candidates.append(stripped_t)
                if abs(line.px_y - top_y_px) <= int(10 * scale_y):
                    top_tier_fragments.append(stripped_t)

        if len(top_tier_fragments) > 1:
            joined_top = " ".join(top_tier_fragments).strip()
            if joined_top not in row.name_candidates:
                row.name_candidates.insert(0, joined_top)

        # Case A: Emoji-only preview row (e.g. Row 2 in screenshot: blank name + '🔥' + '23:23')
        # Vision OCR only sees '23:23' on the right, so `has_left_side_text` is False!
        if not has_left_side_text and row.lines:
            ts_line = row.lines[0]
            ts_cy = int(round(ts_line.center_px[1]))
            # Check white pixels in the name slot at the timestamp's Y level AND 16pt above it
            wpx_same = count_bright_white_pixels_in_name_slot(
                sb_gray, x_name_px, ts_cy, scale_x, scale_y
            )
            wpx_above = count_bright_white_pixels_in_name_slot(
                sb_gray, x_name_px, max(int(8 * scale_y), ts_cy - int(round(16.0 * scale_y))), scale_x, scale_y
            )
            white_px = max(wpx_same, wpx_above)
            row.white_px_in_name_zone = white_px
            if white_px <= 8:
                row.is_blank_or_orphan_preview = True
                row.click_y_log = row.top_y_log
            elif white_px <= 38:
                row.is_dot_candidate = True
                row.click_y_log = row.top_y_log
            elif white_px <= 115:
                row.is_pi_or_symbol_candidate = True
                row.click_y_log = row.top_y_log
            continue

        # Case B: Row has left-side preview text (e.g. 'sent a sticker / 23:07')
        # but Vision OCR missed the tiny symbol ('π' or '.') or blank name above it!
        non_dot_names = [c for c in row.name_candidates if c != "."]
        if not non_dot_names and row.lines:
            top_line = row.lines[0]
            expected_name_y_px = int(round(top_line.center_px[1] - 21.0 * scale_y))
            if expected_name_y_px >= int(8 * scale_y):
                white_px = count_bright_white_pixels_in_name_slot(
                    sb_gray, x_name_px, expected_name_y_px, scale_x, scale_y
                )
                row.white_px_in_name_zone = white_px
                if white_px <= 8:
                    row.is_blank_or_orphan_preview = True
                    row.click_y_log = max(sb_log_y + 18.0, row.top_y_log - 10.0)
                elif white_px <= 38:
                    row.is_dot_candidate = True
                    row.click_y_log = max(sb_log_y + 18.0, row.top_y_log - 10.0)
                elif white_px <= 115:
                    # A tiny 1-char symbol like 'π' above 'sent a sticker' has ~39..115 bright pixels
                    row.is_pi_or_symbol_candidate = True
                    row.click_y_log = max(sb_log_y + 18.0, row.top_y_log - 10.0)

    # Also check 1-row vertical gaps (~115..175pt) for completely un-OCR'd blank/symbol rows
    final_rows: List[ConversationRow] = []
    for i, row in enumerate(raw_rows):
        final_rows.append(row)
        if i + 1 < len(raw_rows):
            gap_log = raw_rows[i + 1].top_y_log - row.top_y_log
            if 115.0 <= gap_log <= 175.0:
                mid_y_log = (row.top_y_log + raw_rows[i + 1].top_y_log) * 0.5
                mid_name_y_px = int(round((mid_y_log - sb_log_y - 10.0) * scale_y))
                white_px = count_bright_white_pixels_in_name_slot(
                    sb_gray, x_name_px, mid_name_y_px, scale_x, scale_y
                )
                band_y1 = max(0, int(round((mid_y_log - sb_log_y - 24.0) * scale_y)))
                band_y2 = min(sb_px_h, int(round((mid_y_log - sb_log_y + 24.0) * scale_y)))
                band_active_px = int(np.sum(sb_gray[band_y1:band_y2, :] > 65))

                if band_active_px > 120 and white_px <= 115:
                    final_rows.append(
                        ConversationRow(
                            top_y_log=mid_y_log,
                            click_y_log=mid_y_log,
                            click_x_log=click_x_log,
                            lines=[],
                            row_date=row.row_date or today,
                            is_today_or_recent=row.is_today_or_recent,
                            is_blank_or_orphan_preview=(white_px <= 8),
                            is_dot_candidate=(8 < white_px <= 38),
                            is_pi_or_symbol_candidate=(38 < white_px <= 115),
                            white_px_in_name_zone=white_px,
                        )
                    )

    final_rows.sort(key=lambda r: r.click_y_log)
    return final_rows


# --- Stage 2: Strict Recipient Matching (Included Names, Unique Symbols, Blank Slot) ---


def score_candidate_against_recipient(
    cand_raw: str, spec: RecipientSpec
) -> Tuple[float, str]:
    if spec.is_invisible:
        return 0.0, ""

    raw_disp = spec.display_name.strip()
    exp_clean = spec.clean_display.strip()
    cand_clean = strip_invisible_and_emojis(cand_raw).strip()

    if not cand_clean:
        return 0.0, ""

    # 1. Unique Symbol 'π' (@paingpaing1123) — matches 'TT', 'IT', '∏', 'pi', 'n', 'r'
    if raw_disp == "π":
        if cand_raw.strip().lower() in PI_OCR_ALIASES or cand_clean.lower() in PI_OCR_ALIASES:
            return 0.97, f"symbol_pi_alias('{cand_raw}')"
        return 0.0, ""

    # 2. Unique Symbol '.' (@user07340968)
    if raw_disp == ".":
        if cand_raw.strip() in (".", "·", "•", ","):
            return 0.97, f"dot_exact('{cand_raw}')"
        return 0.0, ""

    # 3. Exact case-insensitive match (with or without trailing period)
    exp_lower = exp_clean.lower()
    cand_lower = cand_clean.lower()
    if exp_lower == cand_lower or exp_lower.rstrip(".") == cand_lower.rstrip("."):
        return 1.0, f"exact('{cand_clean}')"

    # 4. Short 1-2 character display names ('M', 'K', 'w', 'F.')
    if len(exp_clean) <= 2:
        if exp_clean == "F." and cand_clean.lower() in ("f.", "f"):
            return 0.96, f"short_F('{cand_clean}')"
        if exp_lower == cand_lower:
            return 1.0, f"short_exact('{cand_clean}')"
        return 0.0, ""

    # 5. Short 3-5 character display names ('zari' / 'zarı', 'chl.', 'kise', 'Shaun', 'Cayxl')
    norm_exp = normalize_ocr_confusions(exp_clean)
    norm_cand = normalize_ocr_confusions(cand_clean)
    if norm_exp and norm_cand:
        if norm_exp == norm_cand:
            return 0.96, f"homoglyph_norm('{cand_clean}'~'{exp_clean}')"
        if 3 <= len(norm_exp) <= 5 and abs(len(norm_exp) - len(norm_cand)) <= 1:
            norm_ratio = difflib.SequenceMatcher(None, norm_exp, norm_cand).ratio()
            if norm_ratio >= 0.75:
                return 0.88, f"short_fuzzy({norm_ratio:.2f}:'{cand_clean}')"

    # 6. Multi-word or long display names ('Dalton Maverick', 'AParanoidAndroid', 'WilliamDavidson')
    cand_no_dots = cand_lower.rstrip(".… ")
    if len(cand_no_dots) >= 5 and exp_lower.startswith(cand_no_dots):
        return 0.92, f"truncated_prefix('{cand_clean}')"

    exp_words = [w for w in exp_lower.split() if len(w) >= 5]
    cand_words = [w.rstrip(".…") for w in cand_lower.split()]
    for ew in exp_words:
        if ew in cand_words:
            return 0.91, f"multiword_token('{ew}')"

    if cand_lower.startswith(exp_lower + " ") or (" " + exp_lower) in cand_lower:
        return 0.92, f"substring_token('{cand_clean}')"

    ratio = difflib.SequenceMatcher(None, exp_lower, cand_lower).ratio()
    return ratio, f"fuzzy({ratio:.2f}:'{cand_clean}')"


def match_row_to_pending_recipient(
    row: ConversationRow,
    pending_map: Dict[str, RecipientSpec],
    threshold: float,
) -> Optional[Tuple[RecipientSpec, float, str]]:
    """
    Strictly matches a sidebar row ONLY to recipients present in `pending_map`:
      1. Visible Display Name or unique symbol ('π' -> 'TT', '.' -> '.') match.
      2. Verified Blank-Name slot (ONLY if an invisible 8-space user is still pending).
      3. Verified Dot-Candidate slot (ONLY if '.' is still pending).
      4. Verified Tiny-Symbol slot (ONLY if 'π' or '.' is still pending and Vision OCR
         missed the tiny glyph above the message preview).
    """
    best_spec: Optional[RecipientSpec] = None
    best_score = 0.0
    best_reason = ""

    for cand in row.name_candidates:
        for spec in pending_map.values():
            sc, reason = score_candidate_against_recipient(cand, spec)
            if sc > best_score:
                best_score = sc
                best_spec = spec
                best_reason = reason

    if best_spec is not None and best_score >= threshold:
        return best_spec, best_score, best_reason

    # Blank 8-space slot (e.g. Row 2 with '🔥' and '23:23' and wpx <= 8)
    if row.is_blank_or_orphan_preview and row.is_today_or_recent:
        for spec in pending_map.values():
            if spec.is_invisible:
                return spec, 0.92, f"blank_name_slot(wpx={row.white_px_in_name_zone})"

    # Single Dot '.' slot
    if row.is_dot_candidate and row.is_today_or_recent:
        for spec in pending_map.values():
            if spec.display_name.strip() == ".":
                return spec, 0.90, f"dot_candidate_slot(wpx={row.white_px_in_name_zone})"
        for spec in pending_map.values():
            if spec.display_name.strip() == "π":
                return spec, 0.88, f"small_symbol_slot_for_pi(wpx={row.white_px_in_name_zone})"

    # Tiny Symbol 'π' slot when OCR only saw 'sent a sticker / 23:07'
    if row.is_pi_or_symbol_candidate and row.is_today_or_recent:
        for spec in pending_map.values():
            if spec.display_name.strip() == "π":
                return spec, 0.90, f"pi_symbol_pixel_slot(wpx={row.white_px_in_name_zone})"
        for spec in pending_map.values():
            if spec.display_name.strip() == ".":
                return spec, 0.86, f"symbol_pixel_slot_for_dot(wpx={row.white_px_in_name_zone})"

    return None


# --- Stage 3: Mandatory DM Header Verification (@username + Display Name) ---


def verify_or_identify_recipient_from_header(
    expected_spec: Optional[RecipientSpec],
    pending_map: Dict[str, RecipientSpec],
    hdr_lines: List[OCRLine],
    threshold: float,
) -> Tuple[Optional[RecipientSpec], float, str, str]:
    if not hdr_lines:
        if expected_spec is not None and expected_spec.is_invisible:
            return expected_spec, 0.88, "blank_header_for_invisible_user", "<empty-header>"
        return None, 0.0, "empty_header_ocr", "<empty-header>"

    hdr_summary = " | ".join(l.text for l in hdr_lines)

    raw_tokens: List[str] = []
    explicit_at_tokens: List[str] = []
    for line in hdr_lines:
        for tok in re.split(r"[\s•|]+", line.text.strip()):
            if not tok:
                continue
            raw_tokens.append(tok)
            if tok.startswith("@") and len(tok) >= 3:
                explicit_at_tokens.append(tok.lower().lstrip("@").replace(",", "."))

    # 1. Match against pending @usernames (e.g. @paingpaing1123, @akatsuki7529)
    best_u_spec: Optional[RecipientSpec] = None
    best_u_score = 0.0

    for spec in pending_map.values():
        target_bare = spec.handle_bare.replace(",", ".")
        if not target_bare:
            continue
        for tok in raw_tokens:
            tok_clean = tok.lower().lstrip("@").replace(",", ".")
            if not tok_clean:
                continue
            if tok_clean == target_bare:
                return spec, 1.0, f"header_exact_username({spec.username})", hdr_summary
            if len(target_bare) >= 5 and target_bare in tok_clean:
                sc = 0.96
            else:
                sc = difflib.SequenceMatcher(None, target_bare, tok_clean).ratio()
                if tok.startswith("@") and sc >= 0.78:
                    sc = min(1.0, sc + 0.08)
            if sc > best_u_score:
                best_u_score = sc
                best_u_spec = spec

    if best_u_spec is not None and best_u_score >= max(threshold, 0.80):
        return (
            best_u_spec,
            best_u_score,
            f"header_username({best_u_spec.username}:{best_u_score:.2f})",
            hdr_summary,
        )

    # 2. Reject immediately if a foreign @handle is visible in the header
    if explicit_at_tokens:
        return (
            None,
            best_u_score,
            f"rejected_foreign_handle(@{explicit_at_tokens[0]})",
            hdr_summary,
        )

    # 3. Verify expected_spec's Display Name in the header
    if expected_spec is not None and expected_spec.permanent_id in pending_map:
        for line in hdr_lines:
            sc, reason = score_candidate_against_recipient(line.text, expected_spec)
            if sc >= max(threshold, 0.82):
                return (
                    expected_spec,
                    sc,
                    f"header_display_name({reason})",
                    hdr_summary,
                )

    return None, best_u_score, "header_unverified", hdr_summary


# --- Sent History & Date-Aware Log Store ---


class SentHistoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Any] = {"records": {}, "events": []}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.data["records"] = loaded.get("records", {})
                    self.data["events"] = loaded.get("events", [])
            except Exception:
                pass

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.path)

    def should_skip_recipient(
        self,
        spec: RecipientSpec,
        allow_duplicates: bool,
        today: date,
    ) -> Tuple[bool, str]:
        rec = self.data["records"].get(spec.permanent_id)
        if not isinstance(rec, dict):
            return False, ""

        status = rec.get("status", "")
        rec_date_str = rec.get("local_date", "")
        rec_date: Optional[date] = None
        if rec_date_str:
            try:
                rec_date = date.fromisoformat(rec_date_str)
            except ValueError:
                pass
        if rec_date is None:
            ts = float(rec.get("timestamp_epoch", 0.0))
            if ts > 0:
                rec_date = datetime.fromtimestamp(ts).astimezone().date()

        if status in ("UNCERTAIN", "IN_FLIGHT_TYPING"):
            if rec_date == today:
                return (
                    True,
                    f"Locked as {status} today at {rec.get('timestamp_iso')} "
                    f"({rec.get('reason', 'uncertain send')}). Pass --clear-uncertain to retry.",
                )
            return False, ""

        if not allow_duplicates and status == "SENT":
            if rec_date is not None and (today - rec_date).days < 1:
                return (
                    True,
                    f"Already SENT today ({rec_date.isoformat()}, 1-day interval active)",
                )

        return False, ""

    def update_status(
        self,
        spec: RecipientSpec,
        status: str,
        message: str,
        via_info: str = "",
        header_ocr: str = "",
        reason: str = "",
    ) -> None:
        now_dt = datetime.now(timezone.utc).astimezone()
        today_str = now_dt.date().isoformat()
        entry = {
            "permanent_id": spec.permanent_id,
            "username": spec.username,
            "display_name": spec.display_name,
            "placeholder_name": spec.placeholder_name,
            "status": status,
            "message": message,
            "matched_via": via_info,
            "header_ocr": header_ocr,
            "reason": reason,
            "local_date": today_str,
            "timestamp_epoch": now_dt.timestamp(),
            "timestamp_iso": now_dt.isoformat(timespec="seconds"),
        }
        self.data["records"][spec.permanent_id] = entry
        self.data["events"].append(entry)
        if len(self.data["events"]) > 500:
            self.data["events"] = self.data["events"][-500:]
        self.save()

    def clear_uncertain(self) -> int:
        cleared = 0
        for k, v in list(self.data["records"].items()):
            if isinstance(v, dict) and v.get("status") in (
                "UNCERTAIN",
                "IN_FLIGHT_TYPING",
            ):
                del self.data["records"][k]
                cleared += 1
        if cleared > 0:
            self.save()
        return cleared


# --- Main TikTok DM Automation Engine ---


class TikTokDMAutomationEngine:
    def __init__(
        self,
        config_path: Path,
        nicknames_path: Optional[Path],
        history_path: Path,
        logger: logging.Logger,
        dry_run: bool = False,
        allow_duplicates_override: Optional[bool] = None,
        max_scrolls_override: Optional[int] = None,
        max_chat_age_days: int = 1,
    ) -> None:
        self.config_path = config_path
        self.logger = logger
        self.dry_run = dry_run
        self.max_chat_age_days = max_chat_age_days

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"'{self.config_path}' not found. Run 'python3 setup.py' first."
            )

        with self.config_path.open("r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = json.load(f)

        if allow_duplicates_override is not None:
            self.allow_duplicates = allow_duplicates_override
        else:
            self.allow_duplicates = bool(self.config.get("allow_duplicates", True))

        self.max_scrolls = (
            max_scrolls_override
            if max_scrolls_override is not None
            else int(self.config.get("max_sidebar_scrolls", 5))
        )

        resolved_nicknames = nicknames_path or Path(
            self.config.get("nicknames_file", str(NICKNAMES_DEFAULT_PATH))
        )
        self.all_recipients = parse_nicknames_file(resolved_nicknames)
        self.history = SentHistoryStore(history_path)
        self.safety = SafetyController(logger)

    def read_header_ocr_fast(self) -> List[OCRLine]:
        hdr_reg = self.config["regions"]["recipient_header"]["logical_rect"]
        hx = float(hdr_reg["x"])
        hy = float(hdr_reg["y"])
        hw = max(float(hdr_reg["w"]) + 80.0, 260.0)
        hh = max(float(hdr_reg["h"]) + 34.0, 76.0)
        hdr_img = capture_region_fast_bgr(hx, hy, hw, hh)
        return run_vision_ocr_with_brightness(hdr_img)

    def read_message_input_ocr_fast(self) -> List[OCRLine]:
        inp_reg = self.config["regions"]["message_input"]["logical_rect"]
        ix = float(inp_reg["x"])
        iy = float(inp_reg["y"])
        iw = float(inp_reg["w"])
        ih = float(inp_reg["h"])
        inp_img = capture_region_fast_bgr(ix, iy, iw, ih)
        return run_vision_ocr_with_brightness(inp_img)

    def is_input_box_empty_or_placeholder(
        self, inp_lines: List[OCRLine], sent_text: str
    ) -> Tuple[bool, str]:
        if not inp_lines:
            return True, "<empty>"
        combined = " ".join(l.text for l in inp_lines).strip()
        low = combined.lower()
        placeholders = [
            "send a message",
            "write a message",
            "message...",
            "type a message",
        ]
        if any(ph in low for ph in placeholders):
            return True, combined
        norm_sent = sent_text.strip().lower()
        if norm_sent and norm_sent in low:
            return False, combined
        return True, combined

    def click_verify_and_send(
        self,
        row: ConversationRow,
        expected_spec: RecipientSpec,
        match_reason: str,
        pending_map: Dict[str, RecipientSpec],
        inp_click_x: float,
        inp_click_y: float,
        msg_to_send: str,
        threshold: float,
    ) -> Optional[str]:
        chat_open_wait = 0.30
        input_focus_wait = 0.08
        paste_to_enter_wait = 0.10
        post_send_wait = 0.18

        target_label = f"{expected_spec.placeholder_name} ({expected_spec.username})"

        if self.dry_run:
            self.logger.info(
                f"  [DRY-RUN] Would click row at ({row.click_x_log:.1f}, {row.click_y_log:.1f}) "
                f"for {target_label} [{match_reason}] and verify DM header."
            )
            pending_map.pop(expected_spec.permanent_id, None)
            return "DRY_RUN"

        # Step 1: Click sidebar conversation row
        quartz_fast_click(row.click_x_log, row.click_y_log, self.safety)
        self.safety.sleep_fast(chat_open_wait)

        # Step 2: Mandatory DM Header verification before touching the input box
        hdr_lines = self.read_header_ocr_fast()
        verified_spec, hdr_score, hdr_method, hdr_summary = (
            verify_or_identify_recipient_from_header(
                expected_spec,
                pending_map,
                hdr_lines,
                threshold,
            )
        )

        if verified_spec is None:
            self.safety.sleep_fast(0.20)
            hdr_lines = self.read_header_ocr_fast()
            verified_spec, hdr_score, hdr_method, hdr_summary = (
                verify_or_identify_recipient_from_header(
                    expected_spec,
                    pending_map,
                    hdr_lines,
                    threshold,
                )
            )

        if verified_spec is None:
            self.logger.info(
                f"  [HEADER REJECTED] Row ({row.click_x_log:.1f}, {row.click_y_log:.1f}) "
                f"[{match_reason}] -> Header OCR: '{hdr_summary}' ({hdr_method}). Skipping without clicking input."
            )
            return None

        self.logger.info(
            f"  [HEADER VERIFIED] {verified_spec.placeholder_name} ({verified_spec.username}) "
            f"via {hdr_method} | Header OCR: '{hdr_summary}'"
        )

        # Step 3: Lock as IN_FLIGHT_TYPING before pasting
        self.history.update_status(
            verified_spec,
            "IN_FLIGHT_TYPING",
            msg_to_send,
            via_info=f"{match_reason} -> {hdr_method}",
            header_ocr=hdr_summary,
            reason="Clicked input box and pasting message",
        )

        try:
            quartz_fast_click(inp_click_x, inp_click_y, self.safety)
            self.safety.sleep_fast(input_focus_wait)

            set_clipboard_text(msg_to_send)
            quartz_cmd_v_paste(self.safety)
            self.safety.sleep_fast(paste_to_enter_wait)

            quartz_press_return(self.safety)
            self.safety.sleep_fast(post_send_wait)

            # Step 4: Confirm message input cleared
            inp_lines_after = self.read_message_input_ocr_fast()
            cleared, inp_after_text = self.is_input_box_empty_or_placeholder(
                inp_lines_after, msg_to_send
            )

            if not cleared:
                quartz_press_return(self.safety)
                self.safety.sleep_fast(0.22)
                inp_lines_after = self.read_message_input_ocr_fast()
                cleared, inp_after_text = self.is_input_box_empty_or_placeholder(
                    inp_lines_after, msg_to_send
                )

            if not cleared:
                self.logger.warning(
                    f"  [UNCERTAIN SEND] Composer did not clear for {verified_spec.username} "
                    f"(input OCR='{inp_after_text}'). Locking as UNCERTAIN."
                )
                self.history.update_status(
                    verified_spec,
                    "UNCERTAIN",
                    msg_to_send,
                    via_info=f"{match_reason} -> {hdr_method}",
                    header_ocr=hdr_summary,
                    reason=f"Input box did not clear after Return ('{inp_after_text}')",
                )
                pending_map.pop(verified_spec.permanent_id, None)
                return "UNCERTAIN"

            self.history.update_status(
                verified_spec,
                "SENT",
                msg_to_send,
                via_info=f"{match_reason} -> {hdr_method}",
                header_ocr=hdr_summary,
                reason="Header verified and input box confirmed cleared",
            )
            pending_map.pop(verified_spec.permanent_id, None)
            self.logger.info(
                f"  [SENT CONFIRMED] {verified_spec.placeholder_name} ({verified_spec.username}) "
                f"[{match_reason} -> {hdr_method}]"
            )
            return "SENT"

        except Exception as exc:
            self.logger.error(
                f"  [SEND INTERRUPTED] Error while sending to {verified_spec.username}: {exc}. "
                "Marked UNCERTAIN."
            )
            self.history.update_status(
                verified_spec,
                "UNCERTAIN",
                msg_to_send,
                via_info=f"{match_reason} -> {hdr_method}",
                header_ocr=hdr_summary,
                reason=f"Exception during send: {exc}",
            )
            raise

    def log_diagnostic_summary(
        self,
        today: date,
        pending_map: Dict[str, RecipientSpec],
        unmatched_today_rows: Dict[int, List[str]],
        archived_rows_skipped: int,
    ) -> None:
        if not pending_map:
            return

        self.logger.warning("=" * 72)
        self.logger.warning(
            f"[DIAGNOSTIC REPORT | Date: {today.isoformat()}] "
            f"UNFOUND RECIPIENTS ({len(pending_map)} remaining):"
        )
        for idx, spec in enumerate(pending_map.values(), start=1):
            self.logger.warning(
                f"  {idx:2d}. Display: {spec.placeholder_name!r:<26} | "
                f"ID: {spec.username:<20} | Category: {spec.debug_category}"
            )

        if unmatched_today_rows:
            self.logger.warning("-" * 72)
            self.logger.warning(
                f"[UNCLICKED SIDEBAR ROWS FROM TODAY ({today.isoformat()})]:"
            )
            for seg_no, rows_list in unmatched_today_rows.items():
                self.logger.warning(f"  Segment {seg_no}: {rows_list}")

        if archived_rows_skipped > 0:
            self.logger.info(
                f"(Ignored {archived_rows_skipped} archived sidebar row(s) older than "
                f"{self.max_chat_age_days}d.)"
            )
        self.logger.warning("=" * 72)

    def run(self) -> int:
        start_wall = time.perf_counter()
        today = datetime.now().astimezone().date()
        browser_app = self.config.get("browser_app", "Google Chrome")
        msg_to_send = str(self.config.get("message", "🔥"))
        threshold = float(self.config.get("verification_threshold", 0.80))

        regions = self.config.get("regions", {})
        for req in ("dm_sidebar", "recipient_header", "message_input"):
            if not isinstance(regions.get(req), dict):
                raise RuntimeError(
                    f"Missing '{req}' in config.json. Run 'python3 setup.py'."
                )

        sb_log = regions["dm_sidebar"]["logical_rect"]
        inp_log = regions["message_input"]["logical_rect"]
        inp_click_x = float(inp_log["x"]) + float(inp_log["w"]) * 0.28
        inp_click_y = float(inp_log["y"]) + float(inp_log["h"]) * 0.50

        mode_label = (
            "TEST MODE ON (Allow Duplicates)"
            if self.allow_duplicates
            else f"1-DAY INTERVAL MODE ON (Current Date: {today.isoformat()})"
        )
        self.logger.info(f"Starting TikTok DM Automation | {mode_label}")

        pending_map: Dict[str, RecipientSpec] = {}
        skipped_count = 0
        for spec in self.all_recipients:
            skip, reason = self.history.should_skip_recipient(
                spec, allow_duplicates=self.allow_duplicates, today=today
            )
            if skip:
                self.logger.info(
                    f"[SKIP] {spec.placeholder_name} ({spec.username}): {reason}"
                )
                skipped_count += 1
            else:
                pending_map[spec.permanent_id] = spec

        if not pending_map:
            self.logger.info(
                f"All recipients already processed for {today.isoformat()}! Exiting immediately."
            )
            return 0

        if not ensure_chrome_and_tiktok_messages_tab(self.logger, browser_app):
            return 1

        prior_clipboard = get_clipboard_text()
        if not self.dry_run:
            set_clipboard_text(msg_to_send)

        self.safety.start()
        sent_count = 0
        uncertain_count = 0
        archived_rows_skipped = 0
        unmatched_today_rows: Dict[int, List[str]] = {}
        tested_blank_or_dot_coords: Set[Tuple[int, int]] = set()

        try:
            if not self.dry_run:
                sb_center_x = float(sb_log["x"]) + float(sb_log["w"]) * 0.50
                sb_center_y = float(sb_log["y"]) + float(sb_log["h"]) * 0.50
                quartz_scroll_sidebar_to_top(sb_center_x, sb_center_y, self.safety)
                self.safety.sleep_fast(0.35)

            prev_seg_sig = ""

            for seg_idx in range(1, self.max_scrolls + 2):
                if not pending_map:
                    break

                self.safety.check_or_raise()
                if not is_browser_frontmost_fast(browser_app):
                    ensure_chrome_and_tiktok_messages_tab(self.logger, browser_app)

                sb_img = capture_region_fast_bgr(
                    sb_log["x"], sb_log["y"], sb_log["w"], sb_log["h"]
                )
                sb_px_h, sb_px_w = sb_img.shape[:2]
                scale_x = sb_px_w / float(sb_log["w"])
                scale_y = sb_px_h / float(sb_log["h"])

                sb_lines = run_vision_ocr_with_brightness(sb_img)
                seg_sig = "|".join(f"{l.text}@{l.px_y // 14}" for l in sb_lines)

                if seg_idx > 1 and seg_sig == prev_seg_sig:
                    self.logger.info(
                        f"Reached bottom of DM sidebar at segment {seg_idx}. Stopping."
                    )
                    break
                prev_seg_sig = seg_sig

                rows = analyze_sidebar_segment_rows(
                    sb_img,
                    sb_lines,
                    float(sb_log["x"]),
                    float(sb_log["y"]),
                    float(sb_log["w"]),
                    scale_x,
                    scale_y,
                    today=today,
                    max_chat_age_days=self.max_chat_age_days,
                )

                today_rows_count = sum(1 for r in rows if r.is_today_or_recent)
                old_rows_count = len(rows) - today_rows_count
                archived_rows_skipped += old_rows_count

                self.logger.info(
                    f"[Segment {seg_idx}/{self.max_scrolls + 1}] {len(rows)} row(s) "
                    f"({today_rows_count} current/today, {old_rows_count} archived) | "
                    f"{len(pending_map)} recipient(s) pending"
                )

                for row in rows:
                    if not pending_map:
                        break
                    self.safety.check_or_raise()

                    matched = match_row_to_pending_recipient(
                        row, pending_map, threshold
                    )
                    if matched is None:
                        if row.is_today_or_recent:
                            unmatched_today_rows.setdefault(seg_idx, []).append(
                                row.raw_summary()
                            )
                        continue

                    spec, sc, match_reason = matched

                    if (
                        row.is_blank_or_orphan_preview
                        or row.is_dot_candidate
                        or row.is_pi_or_symbol_candidate
                    ) and not row.name_candidates:
                        slot_key = (seg_idx, int(round(row.click_y_log / 18.0)))
                        if slot_key in tested_blank_or_dot_coords:
                            continue
                        tested_blank_or_dot_coords.add(slot_key)

                    self.logger.info(
                        f"  [MATCH] Row ({row.click_x_log:.1f}, {row.click_y_log:.1f}) "
                        f"-> {spec.placeholder_name} ({spec.username}) via {match_reason} (score={sc:.2f})"
                    )
                    res = self.click_verify_and_send(
                        row=row,
                        expected_spec=spec,
                        match_reason=match_reason,
                        pending_map=pending_map,
                        inp_click_x=inp_click_x,
                        inp_click_y=inp_click_y,
                        msg_to_send=msg_to_send,
                        threshold=threshold,
                    )
                    if res == "SENT":
                        sent_count += 1
                    elif res == "UNCERTAIN":
                        uncertain_count += 1

                if not pending_map:
                    self.logger.info("All pending recipients verified and messaged! Exiting.")
                    break

                if seg_idx >= 2 and today_rows_count == 0 and old_rows_count >= 3:
                    self.logger.info(
                        f"Segment {seg_idx} contains only archived chats older than "
                        f"{self.max_chat_age_days} day(s). Stopping downward scroll."
                    )
                    break

                if seg_idx <= self.max_scrolls:
                    scroll_x = float(sb_log["x"]) + float(sb_log["w"]) * 0.50
                    scroll_y = float(sb_log["y"]) + float(sb_log["h"]) * 0.50
                    quartz_scroll_segment_down(
                        scroll_x, scroll_y, float(sb_log["h"]), self.safety
                    )
                    self.safety.sleep_fast(0.42)

        except InterruptedError as stop_exc:
            self.logger.critical(f"Stopped immediately: {stop_exc}")
            return 2
        finally:
            self.safety.stop()
            if not self.dry_run:
                self.history.save()
                if prior_clipboard is not None:
                    try:
                        set_clipboard_text(prior_clipboard)
                    except Exception:
                        pass
            elapsed = time.perf_counter() - start_wall
            self.logger.info(
                f"Completed in {elapsed:.1f}s | Sent: {sent_count} | "
                f"Uncertain: {uncertain_count} | Skipped: {skipped_count} | "
                f"Unfound: {len(pending_map)}"
            )
            if pending_map:
                self.log_diagnostic_summary(
                    today,
                    pending_map,
                    unmatched_today_rows,
                    archived_rows_skipped,
                )

        return 0


def toggle_allow_duplicates_in_config(config_path: Path) -> bool:
    if not config_path.exists():
        raise FileNotFoundError(f"'{config_path}' not found.")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    current = bool(cfg.get("allow_duplicates", True))
    new_val = not current
    cfg["allow_duplicates"] = new_val
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return new_val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="macOS TikTok DM Automation with Strict Matching & Instant Mouse/ESC Stop."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT_PATH)
    parser.add_argument("--nicknames", type=Path, default=None)
    parser.add_argument("--history", type=Path, default=HISTORY_DEFAULT_PATH)
    parser.add_argument("--log-file", type=Path, default=LOG_DEFAULT_PATH)
    parser.add_argument("--dry-run", action="store_true")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--test",
        "--allow-duplicates",
        "--force",
        dest="allow_duplicates_flag",
        action="store_true",
        help="Send every time you run (ignores 1-day interval check).",
    )
    mode_group.add_argument(
        "--one-day",
        "--no-duplicates",
        dest="one_day_flag",
        action="store_true",
        help="Enforce strict 1-day calendar date interval (skip anyone already messaged today).",
    )
    parser.add_argument(
        "--toggle-test",
        action="store_true",
        help="Flip allow_duplicates ON/OFF in config.json and exit.",
    )
    parser.add_argument(
        "--clear-uncertain",
        action="store_true",
        help="Clear UNCERTAIN / IN_FLIGHT_TYPING locks in sent_history.json before running.",
    )
    parser.add_argument(
        "--max-scrolls",
        type=int,
        default=None,
        help="Maximum sidebar segment scrolls (default: 5).",
    )
    parser.add_argument(
        "--max-chat-age-days",
        type=int,
        default=1,
        help="Maximum age in days for active sidebar rows before treating as archived (default: 1).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.log_file, verbose=args.verbose)

    if args.clear_uncertain:
        store = SentHistoryStore(args.history)
        cleared = store.clear_uncertain()
        logger.info(f"Cleared {cleared} UNCERTAIN/IN_FLIGHT_TYPING lock(s) in {args.history}")

    if args.toggle_test:
        try:
            new_state = toggle_allow_duplicates_in_config(args.config)
            state_str = (
                "ON (Sends every time / duplicates allowed)"
                if new_state
                else "OFF (Strict 1-day date interval check active)"
            )
            logger.info(f"Toggled allow_duplicates in {args.config} -> {state_str}")
            return 0
        except Exception as exc:
            logger.error(f"Could not toggle config: {exc}")
            return 1

    override: Optional[bool] = None
    if args.allow_duplicates_flag:
        override = True
    elif args.one_day_flag:
        override = False

    try:
        engine = TikTokDMAutomationEngine(
            config_path=args.config,
            nicknames_path=args.nicknames,
            history_path=args.history,
            logger=logger,
            dry_run=args.dry_run,
            allow_duplicates_override=override,
            max_scrolls_override=args.max_scrolls,
            max_chat_age_days=args.max_chat_age_days,
        )
        return engine.run()
    except Exception as exc:
        logger.error(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
