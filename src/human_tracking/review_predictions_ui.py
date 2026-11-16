"""
review_predictions_ui.py

Visual review of head-visibility MISCLASSIFICATIONS, for finding wrong annotations.

Workflow: run it on test/ first, fix labels, then val/, then train/.

  1. "Load Dir"      pick a split folder (train/ val/ test/ of dataset_classification)
  2. "Run Inference" delete old *_predictions.json in --predictions_dir, then classify
                     every image and store the fresh result there
  3. review          "Errors only" (default): each tab holds images PREDICTED as that
                     class but labelled something else.
                     Unchecked: every image is shown and tabs group by CURRENT LABEL,
                     so each tab is one class folder to audit end to end. Errors get an
                     orange caption, agreeing crops a grey one; least-confident first.
  4. press 1/2/3     set the true class of the selected image(s); the file is MOVED
                     into that class folder inside input_dir

Images are sorted most-confident-error first, because a model that is confident
and disagrees with the label is the classic sign of a mislabelled image.

Input layout (flat, as produced by split_classification_dataset.py):
    input_dir/
        head_and_shoulders_visible/     *.png
        head_clearly_visible/           *.png
        head_horizontal_or_under_water/ *.png

Keyboard:
    1 / 2 / 3     set true class of selection
    4 / Delete    drop the selection from the dataset (too blurry/partial to judge).
                  Files move to <input_dir>/../_deleted/<split>/<class>/, a sibling of
                  train/val/test, so ImageFolder ignores them but they stay recoverable.
    Left / Right  move selection
    Ctrl+Click    add/remove from selection
    Shift+Click   select range
    Escape        clear selection

Usage:
    python review_predictions_ui.py
    python review_predictions_ui.py --input_dir path/to/val --checkpoint models/x.pth
"""

import argparse
import json
import logging
import math
import os
import queue
import shutil
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from dotenv import load_dotenv
from PIL import Image, ImageTk

# Must happen before HEAD_VISIBILITY_IMG_SIZE is read: inferring at a different
# resolution than the model was trained at silently produces garbage predictions.
load_dotenv()

try:
    from human_tracking.head_visibility_classes import (
        CLASS_COLORS_HEX, CLASS_SHORT_NAMES, TRAINING_CLASSES)
    from human_tracking.classify_human_ui import _fit_square, _get_largest_monitor_rect
except ImportError:
    from head_visibility_classes import CLASS_COLORS_HEX, CLASS_SHORT_NAMES, TRAINING_CLASSES
    from classify_human_ui import _fit_square, _get_largest_monitor_rect

logger = logging.getLogger(__name__)

CLASSES = [c.value for c in TRAINING_CLASSES]          # dataset folder names
# shared with the annotated video drawn by tracking_humans_base, so a class means
# the same colour in the review UI and in the saved video
SHORT = CLASS_SHORT_NAMES
COLOR = CLASS_COLORS_HEX

THUMB, PAD = 240, 8
SORT_KEYS = ["Suspicion", "Name", "Size", "Date"]
CONFIG_FILE = Path.home() / ".review_predictions_ui.json"

DEFAULT_INPUT = r"path/to/data"
DEFAULT_PREDS = r"path/to/data"
DEFAULT_CKPT = r"models\head_visibility.pth"


# ----------------------------------------------------------------- settings

def _load_settings() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_settings(d: dict) -> None:
    try:
        CONFIG_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except OSError:
        pass


def _pred_file(predictions_dir: Path, input_dir: Path) -> Path:
    """One predictions file per split, named after the split folder and its parent."""
    return predictions_dir / f"{input_dir.parent.name}_{input_dir.name}_predictions.json"


# ----------------------------------------------------------------- inference

def run_inference(input_dir: Path, ckpt: Path, model_name: str, img_size: int,
                  progress: queue.Queue) -> dict:
    """Classify every image under input_dir/<class>/. Runs on a worker thread."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets

    try:
        from human_tracking import train_classification as tc
    except ImportError:
        import train_classification as tc

    progress.put(("status", "loading model..."))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = datasets.ImageFolder(str(input_dir), transform=tc._val_tf(img_size))
    tc._drop_corrupt_samples(ds, input_dir.name)
    if not ds.samples:
        raise RuntimeError(f"No readable images under {input_dir}")

    model = tc.build_model(model_name, len(ds.classes)).to(dev)
    model.load_state_dict(torch.load(str(ckpt), map_location=dev, weights_only=True))
    model.eval()

    # num_workers=0: extra worker processes each import torch and can exhaust the
    # Windows paging file, and this also has to stay safe inside a UI thread.
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    total = len(ds.samples)
    done = 0
    items = []
    with torch.no_grad():
        for x, y in loader:
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                out = model(x.to(dev))
            probs = torch.softmax(out.float(), 1).cpu().numpy()
            for i in range(len(y)):
                path, label = ds.samples[done + i]
                p = probs[i]
                k = int(p.argmax())
                items.append({
                    "name": os.path.basename(path),
                    "true": ds.classes[int(label)],
                    "pred": ds.classes[k],
                    "conf": float(p[k]),
                })
            done += len(y)
            progress.put(("progress", (done, total)))

    return {
        "input_dir": str(input_dir),
        "checkpoint": str(ckpt),
        "model": model_name,
        "img_size": img_size,
        "classes": list(ds.classes),
        "items": items,
    }


# ----------------------------------------------------------------- UI

class ReviewUI(tk.Tk):
    _RENDER_BUFFER = 2
    _CACHE_LIMIT = 300

    def __init__(self, args):
        super().__init__()
        self.title("Prediction Review")

        s = _load_settings()
        self.input_dir = Path(args.input_dir or s.get("input_dir") or DEFAULT_INPUT)
        self.predictions_dir = Path(args.predictions_dir or s.get("predictions_dir") or DEFAULT_PREDS)
        self.checkpoint = Path(args.checkpoint or s.get("checkpoint") or DEFAULT_CKPT)
        self.model_name = args.model or s.get("model") or "efficientnet_v2_s"
        self.img_size = args.img_size or s.get("img_size") or int(os.getenv("HEAD_VISIBILITY_IMG_SIZE", "224"))

        self.items: list[dict] = []        # only misclassified, sorted by confidence desc
        self.filtered: list[int] = []
        self.selected: set[int] = set()
        self.last_fi: int | None = None
        self.anchor_fi: int | None = None
        self.filter: str | None = s.get("filter")
        self._errors_only = tk.BooleanVar(value=s.get("errors_only", True))
        self._sort_key = tk.StringVar(value=s.get("sort_key", "Suspicion"))
        self._sort_desc = tk.BooleanVar(value=s.get("sort_desc", False))
        # last selected image per split folder, so test/val/train each resume where you left off
        self._last_items: dict[str, str] = dict(s.get("last_items", {}))
        self._thumb = int(s.get("thumb", THUMB))
        self._cache: dict[int, ImageTk.PhotoImage] = {}
        self._redraw_job = None
        self._pending_scroll = None
        self._q: queue.Queue = queue.Queue()
        self._busy = False

        mon = _get_largest_monitor_rect()
        if mon:
            left, top, w, h = mon
            self.geometry(f"{w}x{h}+{left}+{top}")
        self.minsize(900, 500)
        self.state("zoomed")

        self._build_ui()
        self._load_predictions()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------ build

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        bar = tk.Frame(self, bd=1, relief=tk.SUNKEN)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        tk.Button(bar, text="Load Dir", command=self._load_dir,
                  font=("Arial", 10, "bold"), padx=10, pady=4).pack(side=tk.LEFT, padx=4)
        self._infer_btn = tk.Button(bar, text="Run Inference", command=self._run_inference,
                                    font=("Arial", 10, "bold"), padx=10, pady=4, bg="#1976d2", fg="white")
        self._infer_btn.pack(side=tk.LEFT, padx=4)
        self._dir_var = tk.StringVar()
        tk.Label(bar, textvariable=self._dir_var, font=("Arial", 9), fg="#555").pack(side=tk.LEFT, padx=8)
        self._status = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._status, font=("Arial", 9), fg="#1976d2").pack(side=tk.RIGHT, padx=10)
        self._pbar = ttk.Progressbar(bar, mode="determinate", length=260)
        self._pbar.pack(side=tk.RIGHT, padx=(10, 2), pady=4)

        frow = tk.Frame(self, bg="#222")
        frow.grid(row=1, column=0, columnspan=2, sticky="ew")
        self._filter_btns = {}
        for key, color in [(None, "#444")] + [(c, COLOR[c]) for c in CLASSES]:
            b = tk.Button(frow, font=("Arial", 9, "bold"), bg=color, fg="white",
                          relief=tk.FLAT, padx=10, pady=5, command=lambda k=key: self._set_filter(k))
            b.pack(side=tk.LEFT, padx=2, pady=4)
            self._filter_btns[key] = b
        tk.Checkbutton(frow, text="Errors only", variable=self._errors_only, command=self._on_errors_toggle,
                       bg="#222", fg="white", selectcolor="#444", activebackground="#222",
                       activeforeground="white", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(14, 4))
        tk.Label(frow, text="Sort:", bg="#222", fg="#aaa",
                 font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(14, 2))
        sort_box = ttk.Combobox(frow, textvariable=self._sort_key, values=SORT_KEYS,
                                state="readonly", width=10)
        sort_box.pack(side=tk.LEFT, padx=2)
        sort_box.bind("<<ComboboxSelected>>", lambda _: self._apply_sort())
        tk.Checkbutton(frow, text="Descending", variable=self._sort_desc,
                       command=self._apply_sort,
                       bg="#222", fg="white", selectcolor="#444", activebackground="#222",
                       activeforeground="white", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(6, 4))
        self._update_tab_labels()
        self._count_var = tk.StringVar()
        tk.Label(frow, textvariable=self._count_var, bg="#222", fg="#aaa",
                 font=("Arial", 9)).pack(side=tk.RIGHT, padx=12)
        tk.Label(frow, text="Size:", bg="#222", fg="#aaa", font=("Arial", 9)).pack(side=tk.RIGHT, padx=(12, 2))
        self._size_var = tk.IntVar(value=self._thumb)
        tk.Scale(frow, variable=self._size_var, from_=80, to=400, resolution=20,
                 orient=tk.HORIZONTAL, bg="#222", fg="white", highlightthickness=0,
                 troughcolor="#555", length=130, command=self._on_size).pack(side=tk.RIGHT, padx=(0, 4))

        cf = tk.Frame(self)
        cf.grid(row=2, column=0, sticky="nsew", padx=(4, 0), pady=4)
        cf.columnconfigure(0, weight=1)
        cf.rowconfigure(0, weight=1)
        self._canvas = tk.Canvas(cf, bg="#1a1a1a", highlightthickness=0)
        self._vs = tk.Scrollbar(cf, orient=tk.VERTICAL, command=self._on_scrollbar)
        self._canvas.configure(yscrollcommand=self._on_yscroll)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._vs.grid(row=0, column=1, sticky="ns")
        self._canvas.bind("<Configure>", self._on_configure)
        self._canvas.bind("<Button-1>", self._on_click)
        self.bind("<MouseWheel>", lambda e: (self._canvas.yview_scroll(-(e.delta // 120), "units"),
                                             self._schedule_redraw()))
        self.bind("<Left>", lambda _: self._move(-1))
        self.bind("<Right>", lambda _: self._move(1))
        self.bind("<Escape>", lambda _: self._clear_sel())
        for i, c in enumerate(CLASSES, 1):
            self.bind(str(i), lambda _, cc=c: self._reassign(cc))
        self.bind(str(len(CLASSES) + 1), lambda _: self._delete_selected())
        self.bind("<Delete>", lambda _: self._delete_selected())

        rp = tk.Frame(self, bg="#2a2a2a", width=230)
        rp.grid(row=2, column=1, sticky="ns", padx=4, pady=4)
        rp.grid_propagate(False)
        tk.Label(rp, text="Set TRUE class to:", bg="#2a2a2a", fg="#aaa",
                 font=("Arial", 9, "bold")).pack(pady=(14, 4), padx=10)
        self._sel_var = tk.StringVar(value="Nothing selected")
        tk.Label(rp, textvariable=self._sel_var, bg="#2a2a2a", fg="white",
                 font=("Arial", 9), wraplength=210).pack(pady=(0, 12), padx=10)
        for i, c in enumerate(CLASSES, 1):
            tk.Button(rp, text=f"{i}. {SHORT[c]}", bg=COLOR[c], fg="white",
                      font=("Arial", 10, "bold"), padx=6, pady=10, relief=tk.FLAT,
                      wraplength=200, command=lambda cc=c: self._reassign(cc)).pack(fill=tk.X, padx=8, pady=3)
        tk.Label(rp, text="Moves the file into that\nclass folder in input_dir.",
                 bg="#2a2a2a", fg="#777", font=("Arial", 8)).pack(pady=(10, 0), padx=8)

        tk.Frame(rp, bg="#444", height=2).pack(fill=tk.X, padx=8, pady=(16, 10))
        tk.Button(rp, text=f"{len(CLASSES) + 1}. Delete  (unusable)", bg="#5d4037", fg="white",
                  font=("Arial", 10, "bold"), padx=6, pady=10, relief=tk.FLAT,
                  wraplength=200, command=self._delete_selected).pack(fill=tk.X, padx=8, pady=3)
        tk.Label(rp, text="For crops too blurry/partial to\njudge. Removed from the dataset,\n"
                          "moved to _deleted/ next to the\nsplits so you can still recover it.",
                 bg="#2a2a2a", fg="#777", font=("Arial", 8), justify=tk.LEFT).pack(pady=(6, 0), padx=8)

        self._info = tk.StringVar(value="Load a directory, then Run Inference.")
        tk.Label(self, textvariable=self._info, font=("Arial", 9), fg="#666",
                 anchor=tk.W).grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(2, 6))

    # ------------------------------------------------------------ data

    def _load_predictions(self):
        """Build the item list from input_dir, then attach predictions where present.

        input_dir/<class>/ is the source of truth for which images exist and what they
        are labelled; the predictions file only adds the model's opinion. So all-images
        review works without ever running inference, and a stale or missing predictions
        file can never hide or resurrect images.
        """
        self._dir_var.set(str(self.input_dir))

        on_disk = []                                    # (name, label), CLASSES order
        for c in CLASSES:
            d = self.input_dir / c
            if d.is_dir():
                on_disk.extend((p.name, c) for p in sorted(d.glob("*.png")))

        pf = _pred_file(self.predictions_dir, self.input_dir)
        preds, ckpt = {}, None
        if pf.is_file():
            try:
                data = json.loads(pf.read_text(encoding="utf-8"))
                preds = {r["name"]: r for r in data.get("items", [])}
                ckpt = Path(data.get("checkpoint", "?")).name
            except (OSError, ValueError) as e:
                messagebox.showerror("Predictions", f"Cannot read {pf}:\n{e}")

        # A crop duplicated across class folders appears once per copy on disk. Show it
        # once; assigning a class later collapses the copies.
        seen, items, dups = set(), [], 0
        for name, label in on_disk:
            if name in seen:
                dups += 1
                continue
            seen.add(name)
            r = preds.get(name)
            items.append({"name": name, "true": label,
                          "pred": r["pred"] if r else None,
                          "conf": r["conf"] if r else None})

        self.items = items
        self._apply_sort(refilter=False)

        n_pred = sum(1 for it in items if it["pred"] is not None)
        n_err = sum(1 for it in items if it["pred"] is not None and it["pred"] != it["true"])
        if n_pred:
            acc = 100 * (1 - n_err / n_pred)
            msg = f"{len(items)} images, {n_err} errors (acc {acc:.2f}%) | ckpt {ckpt}"
            if n_pred < len(items):
                msg += f"  |  {len(items) - n_pred} without prediction (Run Inference to include)"
        else:
            msg = (f"{len(items)} images in {self.input_dir.name}, no predictions "
                   f"— uncheck 'Errors only' to review them, or Run Inference to find errors")
        self._info.set(msg + (f"  |  {dups} duplicate(s) hidden" if dups else ""))
        if dups:
            logger.warning("%d crop(s) appear in more than one class folder; "
                           "assign a class to collapse them.", dups)
        self._set_filter(self.filter)
        self._restore_position()

    def _save_predictions(self, drop: set | None = None):
        """Persist current true-labels so fixes survive a restart.

        `drop` names records to remove outright (deleted crops) — leaving them in
        would resurrect them as unloadable thumbnails on the next reload.
        """
        pf = _pred_file(self.predictions_dir, self.input_dir)
        if not pf.is_file():
            return
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if drop:
            data["items"] = [it for it in data.get("items", []) if it["name"] not in drop]
        by_name = {it["name"]: it for it in self.items}
        for it in data.get("items", []):
            cur = by_name.get(it["name"])
            if cur is not None:
                it["true"] = cur["true"]
        try:
            pf.write_text(json.dumps(data), encoding="utf-8")
        except OSError as e:
            logger.warning("Could not save predictions: %s", e)

    def _load_dir(self):
        d = filedialog.askdirectory(title="Select split folder (train/ val/ test/)",
                                    initialdir=str(self.input_dir.parent if self.input_dir.exists() else "/"))
        if not d:
            return
        p = Path(d)
        missing = [c for c in CLASSES if not (p / c).is_dir()]
        if missing:
            messagebox.showwarning("Wrong folder",
                                   f"{p}\n\nis missing class folders:\n" + "\n".join(missing))
            return
        self._remember_current()               # keep the old split's position
        self.input_dir = p
        self.filter = None
        self._load_predictions()

    # ------------------------------------------------------------ inference

    def _existing_prediction_files(self) -> list[Path]:
        """Prediction files this tool wrote, cleared before each new run.

        Deliberately globs only *_predictions.json rather than wiping the whole
        folder, so a mistyped --predictions_dir can't destroy unrelated data.
        """
        if not self.predictions_dir.is_dir():
            return []
        return sorted(self.predictions_dir.glob("*_predictions.json"))

    def _run_inference(self):
        if self._busy:
            return
        if not self.checkpoint.is_file():
            messagebox.showerror("Checkpoint", f"Not found:\n{self.checkpoint}")
            return
        n = sum(len(list((self.input_dir / c).glob('*.png'))) for c in CLASSES if (self.input_dir / c).is_dir())
        stale = self._existing_prediction_files()
        extra = (f"\n\nThis first DELETES {len(stale)} existing prediction file(s) in\n"
                 f"{self.predictions_dir}") if stale else ""
        if not messagebox.askyesno("Run inference",
                                   f"Classify {n} images in\n{self.input_dir}\n\n"
                                   f"using {self.checkpoint.name}?{extra}"):
            return
        self._busy = True
        self._infer_btn.config(state=tk.DISABLED, text="Running...")
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        for f in stale:
            try:
                f.unlink()
                logger.info("cleared old predictions: %s", f.name)
            except OSError as e:
                logger.warning("could not delete %s: %s", f, e)
        self._remember_current()               # the list is about to be replaced
        self.items = []
        self._set_filter(self.filter)
        self._pbar.config(value=0, maximum=max(n, 1))

        def work():
            try:
                res = run_inference(self.input_dir, self.checkpoint, self.model_name,
                                    self.img_size, self._q)
                pf = _pred_file(self.predictions_dir, self.input_dir)
                pf.write_text(json.dumps(res), encoding="utf-8")
                self._q.put(("done", str(pf)))
            except Exception as e:                       # surfaced in the UI thread
                logger.exception("inference failed")
                self._q.put(("error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=work, daemon=True).start()
        self.after(100, self._poll)

    def _poll(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "progress":
                    done, total = payload
                    self._pbar.config(maximum=total, value=done)
                    self._status.set(f"inference {done}/{total} ({100*done/total:.0f}%)")
                elif kind == "status":
                    self._status.set(payload)
                elif kind == "done":
                    self._busy = False
                    self._infer_btn.config(state=tk.NORMAL, text="Run Inference")
                    self._pbar.config(value=self._pbar["maximum"])
                    self._status.set("inference complete")
                    self._load_predictions()
                    return
                elif kind == "error":
                    self._busy = False
                    self._infer_btn.config(state=tk.NORMAL, text="Run Inference")
                    self._pbar.config(value=0)
                    self._status.set("")
                    messagebox.showerror("Inference failed", payload)
                    return
        except queue.Empty:
            pass
        if self._busy:
            self.after(100, self._poll)

    # ------------------------------------------------------------ filter / grid

    def _remember_current(self):
        """Record the selected image for this split. Nothing selected keeps the old value,
        so closing after a relabel (which can leave no selection) doesn't lose your place."""
        fi = self.last_fi if self.last_fi is not None else (min(self.selected) if self.selected else None)
        if fi is not None and 0 <= fi < len(self.filtered):
            self._last_items[str(self.input_dir)] = self.items[self.filtered[fi]]["name"]

    def _restore_position(self):
        """Re-select the remembered image and scroll it into view.

        Right after startup the window isn't mapped yet, so the canvas has no real size
        and any scroll maths would use a placeholder width. In that case the scroll is
        deferred to the first real <Configure>; later loads (Load Dir, inference) have
        a laid-out canvas and scroll immediately.
        """
        name = self._last_items.get(str(self.input_dir))
        if not name:
            return
        fi = next((f for f, idx in enumerate(self.filtered) if self.items[idx]["name"] == name), None)
        if fi is None:
            logger.info("last image %s not in the current tab/filter; starting at the top", name)
            return
        self.selected = {fi}
        self.last_fi = self.anchor_fi = fi
        self._update_sel()
        if self._canvas.winfo_width() > 1:
            self._scroll_to(fi)
        else:
            self._pending_scroll = fi
        self._schedule_redraw()

    def _ensure_meta(self, field: str):
        """Fill image size or file time for every item (needed before sorting on it).

        Only the image header is read, not the pixels, so this stays fast even on train.
        """
        missing = [it for it in self.items if field not in it]
        if not missing:
            return
        self._status.set(f"reading {'image sizes' if field == 'wh' else 'file dates'} "
                         f"for {len(missing)} images...")
        self.update_idletasks()
        for it in missing:
            p = self.input_dir / it["true"] / it["name"]
            try:
                if field == "wh":
                    with Image.open(p) as im:
                        it["wh"] = im.size
                else:
                    it["mtime"] = p.stat().st_mtime
            except OSError:
                it[field] = (0, 0) if field == "wh" else 0.0
        self._status.set("")

    def _apply_sort(self, refilter: bool = True):
        """Reorder self.items by the chosen key; Descending reverses it.

        Suspicion (default): errors by confidence (most confident disagreement =
        likeliest mislabel), then agreeing crops least-confident first, then crops
        with no prediction. Size is pixel area (w*h). Date is the file's modified
        time, which copy2 preserved from when the crop was cut.
        """
        key = self._sort_key.get()
        if key == "Size":
            self._ensure_meta("wh")
            fn = lambda it: (it["wh"][0] * it["wh"][1], it["name"].lower())
        elif key == "Date":
            self._ensure_meta("mtime")
            fn = lambda it: (it["mtime"], it["name"].lower())
        elif key == "Name":
            fn = lambda it: it["name"].lower()
        else:
            def fn(it):
                if it["pred"] is None:
                    return (2, 0.0, it["name"].lower())
                if it["pred"] != it["true"]:
                    return (0, -it["conf"], it["name"].lower())
                return (1, it["conf"], it["name"].lower())
        self.items.sort(key=fn, reverse=self._sort_desc.get())
        if refilter:
            self._refilter(keep_view=False)

    def _visible(self, it: dict) -> bool:
        """Errors-only mode: tabs group by MODEL PREDICTION (what the model thought).
        All mode: tabs group by CURRENT LABEL, so each tab is one class folder to audit."""
        errors_only = self._errors_only.get()
        if errors_only and (it["pred"] is None or it["true"] == it["pred"]):
            return False                       # no prediction can't be an error
        if self.filter is None:
            return True
        return (it["pred"] if errors_only else it["true"]) == self.filter

    def _update_tab_labels(self):
        errors_only = self._errors_only.get()
        self._filter_btns[None].config(text="All errors" if errors_only else "All")
        for i, c in enumerate(CLASSES, 1):
            self._filter_btns[c].config(text=f"{i}. {'pred' if errors_only else 'label'} {SHORT[c]}")

    def _on_errors_toggle(self):
        self._update_tab_labels()
        self._set_filter(self.filter)

    def _set_filter(self, key):
        """Switch tab: rebuild the view and jump to the top."""
        self.filter = key
        for k, b in self._filter_btns.items():
            b.config(relief=tk.RAISED if k == key else tk.FLAT, bd=3 if k == key else 1)
        self._refilter(keep_view=False)

    def _refilter(self, keep_view: bool):
        """Recompute the visible list.

        keep_view=True is used after relabel/delete: during a full-folder review a jump
        back to the top after every keypress would be unusable, so the scroll position
        stays put and the selection moves to whatever now occupies the edited slot.
        """
        slot = min(self.selected) if self.selected else None
        self.filtered = [i for i, it in enumerate(self.items) if self._visible(it)]
        self._count_var.set(f"{len(self.filtered)} images")
        self.selected.clear()
        self.last_fi = self.anchor_fi = None
        self._cache.clear()
        if keep_view:
            if slot is not None and self.filtered:
                fi = min(slot, len(self.filtered) - 1)
                self.selected = {fi}
                self.last_fi = self.anchor_fi = fi
        else:
            self._canvas.yview_moveto(0.0)
        self._update_sel()
        self._schedule_redraw()

    @property
    def _cell_w(self): return self._thumb + PAD * 2

    @property
    def _cell_h(self): return self._thumb + PAD * 2 + 32

    def _on_size(self, _=None):
        self._thumb = self._size_var.get()
        self._cache.clear()
        self._schedule_redraw()

    def _on_configure(self, _=None):
        if self._pending_scroll is not None:
            fi = self._pending_scroll
            self._pending_scroll = None
            self._scroll_to(fi)
        self._schedule_redraw()

    def _on_yscroll(self, *a):
        self._vs.set(*a)
        self._schedule_redraw()

    def _on_scrollbar(self, *a):
        self._canvas.yview(*a)
        self._schedule_redraw()

    def _schedule_redraw(self):
        if self._redraw_job:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(40, self._redraw)

    def _cols(self):
        return max(1, (self._canvas.winfo_width() or 900) // self._cell_w)

    def _path_of(self, idx: int) -> Path:
        it = self.items[idx]
        return self.input_dir / it["true"] / it["name"]

    def _all_copies(self, name: str) -> list[Path]:
        """Every class folder currently holding this crop.

        More than one means the dataset carries contradictory labels for it, so
        relabel and delete both sweep all copies rather than just the one on screen.
        """
        return [self.input_dir / c / name
                for c in CLASSES if (self.input_dir / c / name).is_file()]

    def _trash(self, path: Path) -> None:
        """Move one copy out of the dataset, keeping it recoverable."""
        stale = self._trash_dir() / path.parent.name / path.name
        stale.parent.mkdir(parents=True, exist_ok=True)
        if stale.exists():
            stale.unlink()
        shutil.move(str(path), str(stale))

    def _redraw(self):
        self._redraw_job = None
        self._canvas.delete("all")
        if not self.filtered:
            self._canvas.create_text(320, 60, text=("No misclassified images in this tab"
                                                    if self._errors_only.get() else "No images in this tab"),
                                     fill="#555", font=("Arial", 14))
            self._canvas.config(scrollregion=(0, 0, 640, 160))
            return
        cols = self._cols()
        rows = math.ceil(len(self.filtered) / cols)
        total_h = rows * self._cell_h + PAD * 2
        cw = self._canvas.winfo_width() or 900
        self._canvas.config(scrollregion=(0, 0, cw, total_h))

        top, bot = self._canvas.yview()
        r0 = max(0, int(top * total_h / self._cell_h) - self._RENDER_BUFFER)
        r1 = min(rows - 1, int(bot * total_h / self._cell_h) + self._RENDER_BUFFER)
        fi0, fi1 = r0 * cols, min(len(self.filtered) - 1, (r1 + 1) * cols - 1)

        for fi in range(fi0, fi1 + 1):
            idx = self.filtered[fi]
            it = self.items[idx]
            row, col = divmod(fi, cols)
            x0 = col * self._cell_w + PAD
            y0 = row * self._cell_h + PAD
            self._canvas.create_image(x0, y0, anchor=tk.NW, image=self._thumb_of(idx))
            # border = current annotation
            self._canvas.create_rectangle(x0 - 3, y0 - 3, x0 + self._thumb + 3, y0 + self._thumb + 3,
                                          outline=COLOR[it["true"]], width=4)
            wrong = it["pred"] is not None and it["true"] != it["pred"]
            if it["pred"] is None:
                label = f"now: {SHORT[it['true']]}   |   no prediction"
            elif wrong:
                label = f"now: {SHORT[it['true']]}   |   model: {SHORT[it['pred']]} {it['conf']:.2f}"
            else:
                label = f"now: {SHORT[it['true']]}   |   model agrees {it['conf']:.2f}"
            # in the all-images view the orange caption is how an error stands out
            self._canvas.create_text(x0 + self._thumb // 2, y0 + self._thumb + 6, text=label,
                                     fill="#ffb74d" if wrong else "#9e9e9e",
                                     font=("Arial", 9, "bold"), width=self._thumb, anchor=tk.N)
            wh = it.get("wh")
            # size first: long names wrap, and the size must stay on the visible line
            name_text = f"{wh[0]}x{wh[1]}   {it['name']}" if wh else it["name"]
            self._canvas.create_text(x0 + self._thumb // 2, y0 + self._thumb + 20, text=name_text,
                                     fill="#777", font=("Arial", 7), width=self._thumb, anchor=tk.N)
            if fi in self.selected:
                self._canvas.create_rectangle(x0 - 7, y0 - 7, x0 + self._thumb + 7, y0 + self._thumb + 7,
                                              outline="white", width=5)
        keep = set(self.filtered[max(0, fi0 - 150):fi1 + 150])
        for k in [k for k in self._cache if k not in keep]:
            del self._cache[k]

    def _thumb_of(self, idx: int):
        if idx not in self._cache:
            try:
                img = Image.open(self._path_of(idx))
                # original crop size, kept on the item so it survives cache clears
                self.items[idx]["wh"] = img.size
                pil = _fit_square(img.convert("RGB"), self._thumb)
            except Exception:
                pil = Image.new("RGB", (self._thumb, self._thumb), (60, 30, 30))
            self._cache[idx] = ImageTk.PhotoImage(pil)
        return self._cache[idx]

    def _scroll_to(self, fi):
        cols = self._cols()
        rows = math.ceil(len(self.filtered) / cols)
        total_h = rows * self._cell_h + PAD * 2
        cw = self._canvas.winfo_width() or 900
        self._canvas.config(scrollregion=(0, 0, cw, total_h))
        if total_h <= 0:
            return
        vis = self._canvas.winfo_height() or 600
        centre = (fi // cols) * self._cell_h + PAD + self._cell_h / 2
        self._canvas.yview_moveto(max(0, min(centre - vis / 2, total_h - vis)) / total_h)

    # ------------------------------------------------------------ selection

    def _fi_at(self, x, y):
        cols = self._cols()
        col, row = int(x // self._cell_w), int(y // self._cell_h)
        fi = row * cols + col
        if 0 <= fi < len(self.filtered) and col < cols:
            x0, y0 = col * self._cell_w + PAD, row * self._cell_h + PAD
            if x0 <= x <= x0 + self._thumb and y0 <= y <= y0 + self._thumb + 30:
                return fi
        return None

    def _on_click(self, e):
        fi = self._fi_at(self._canvas.canvasx(e.x), self._canvas.canvasy(e.y))
        if fi is None:
            return
        if (e.state & 0x1) and self.anchor_fi is not None:       # shift
            lo, hi = sorted((self.anchor_fi, fi))
            self.selected = set(range(lo, hi + 1))
        elif e.state & 0x4:                                      # ctrl
            if fi in self.selected:
                self.selected.discard(fi)
            else:
                self.selected.add(fi)
            self.anchor_fi = fi
        else:
            self.selected = {fi}
            self.anchor_fi = fi
        self.last_fi = fi
        self._update_sel()
        self._schedule_redraw()

    def _clear_sel(self):
        self.selected.clear()
        self.last_fi = self.anchor_fi = None
        self._update_sel()
        self._schedule_redraw()

    def _move(self, d):
        if not self.filtered:
            return
        fi = max(0, min(len(self.filtered) - 1, (self.last_fi or 0) + d))
        self.selected = {fi}
        self.last_fi = self.anchor_fi = fi
        self._update_sel()
        self._scroll_to(fi)
        self._schedule_redraw()

    def _update_sel(self):
        n = len(self.selected)
        if n == 0:
            self._sel_var.set("Nothing selected")
        elif n == 1:
            it = self.items[self.filtered[next(iter(self.selected))]]
            model = (f"{SHORT[it['pred']]} ({it['conf']:.3f})" if it["pred"] is not None
                     else "no prediction")
            self._sel_var.set(f"1 selected\nnow: {SHORT[it['true']]}\nmodel: {model}")
        else:
            self._sel_var.set(f"{n} images selected")

    # ------------------------------------------------------------ relabel

    def _reassign(self, new_cls: str):
        if not self.selected:
            return
        idxs = [self.filtered[fi] for fi in sorted(self.selected)]
        moved, failed, resolved = 0, [], 0
        for idx in idxs:
            it = self.items[idx]
            if it["true"] == new_cls:
                continue
            src = self._path_of(idx)
            dst = self.input_dir / new_cls / it["name"]
            try:
                # A crop belongs to exactly one class. Clear out every other copy first
                # (the target slot, or a third class folder) so the contradictory labels
                # are resolved in one action instead of leaving orphans behind.
                for other in self._all_copies(it["name"]):
                    if other == src:
                        continue
                    self._trash(other)
                    resolved += 1
                    logger.warning("duplicate: %s also sat in %s; that copy moved to _deleted/",
                                   it["name"], other.parent.name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            except OSError as e:
                failed.append(f"{it['name']} ({e})")
                continue
            it["true"] = new_cls
            moved += 1
            logger.info("relabel %s -> %s", it["name"], new_cls)

        self._save_predictions()
        # visibility is decided by _visible(): in errors-only mode a crop that now agrees
        # with the model drops out; in all mode it moves to its new label's tab
        self._refilter(keep_view=True)
        self._status.set(f"moved {moved} → {SHORT[new_cls]}"
                         + (f", {resolved} duplicate(s) resolved" if resolved else "")
                         + (f", {len(failed)} failed" if failed else ""))
        if failed:
            messagebox.showwarning("Some moves failed", "\n".join(failed[:12]))

    # ------------------------------------------------------------ delete

    def _trash_dir(self) -> Path:
        """Sibling of train/val/test, so ImageFolder never sees it as a 4th class."""
        return self.input_dir.parent / "_deleted" / self.input_dir.name

    def _delete_selected(self):
        """Remove unjudgeable crops from the dataset (moved to _deleted/, recoverable)."""
        if not self.selected:
            return
        idxs = [self.filtered[fi] for fi in sorted(self.selected)]
        if len(idxs) > 1 and not messagebox.askyesno(
                "Delete images",
                f"Remove {len(idxs)} images from the dataset?\n\n"
                f"They are moved to:\n{self._trash_dir()}"):
            return

        gone, failed = set(), []
        for idx in idxs:
            it = self.items[idx]
            copies = self._all_copies(it["name"]) or [self._path_of(idx)]
            try:
                for p in copies:                 # sweep every class folder, not just this one
                    self._trash(p)
            except OSError as e:
                failed.append(f"{it['name']} ({e})")
                continue
            gone.add(it["name"])
            logger.info("deleted %s (was %s)", it["name"], it["true"])

        self._save_predictions(drop=gone)
        self.items = [it for it in self.items if it["name"] not in gone]
        self._refilter(keep_view=True)
        self._status.set(f"deleted {len(gone)}" + (f", {len(failed)} failed" if failed else ""))
        if failed:
            messagebox.showwarning("Some deletions failed", "\n".join(failed[:12]))

    # ------------------------------------------------------------ close

    def destroy(self):
        # a redraw scheduled via after() would otherwise fire on a dead window
        if self._redraw_job:
            self.after_cancel(self._redraw_job)
            self._redraw_job = None
        super().destroy()

    def _on_close(self):
        self._remember_current()
        _save_settings({
            "input_dir": str(self.input_dir),
            "predictions_dir": str(self.predictions_dir),
            "checkpoint": str(self.checkpoint),
            "model": self.model_name,
            "img_size": self.img_size,
            "filter": self.filter,
            "thumb": self._thumb,
            "last_items": self._last_items,
            "errors_only": self._errors_only.get(),
            "sort_key": self._sort_key.get(),
            "sort_desc": self._sort_desc.get(),
        })
        self.destroy()


def main():
    p = argparse.ArgumentParser(description="Review misclassified head-visibility crops.")
    p.add_argument("--input_dir", default=None, help=f"Split folder. Default: last used, else {DEFAULT_INPUT}")
    p.add_argument("--predictions_dir", default=None, help=f"Where prediction json files go. Default: {DEFAULT_PREDS}")
    p.add_argument("--checkpoint", default=None, help=f"Model checkpoint. Default: {DEFAULT_CKPT}")
    p.add_argument("--model", default=None, help="Backbone name (default efficientnet_v2_s).")
    p.add_argument("--img_size", type=int, default=None, help="Defaults to HEAD_VISIBILITY_IMG_SIZE.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ReviewUI(args).mainloop()


if __name__ == "__main__":
    main()
