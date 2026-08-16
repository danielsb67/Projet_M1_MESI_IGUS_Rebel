#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspection_data.py — Revue humaine des épisodes bruts (validation / rejet)
===========================================================================

Le filtre automatique de l'IHM (success du meta.json) se trompe parfois :
cet outil rejoue chaque cycle (front + wrist côte à côte) et laisse l'HUMAIN
trancher :

  ✓ Garder   → écrit ``human_check: "ok"`` dans meta.json (atomique)
  ✗ Rejeter  → DÉPLACE le dossier épisode vers datasets/raw_echecs/
               (rien n'est supprimé) + ``human_check: "rejected"``
  ↩ Annuler  → ramène le dernier épisode rejeté à sa place

Raccourcis : Espace lecture/pause · ←/→ ±1 frame · Shift+←/→ ±10 ·
             ↑/↓ épisode précédent/suivant · V garder · X rejeter.

⚠ Hardlinks : raw_v2_1 partage ses fichiers avec raw_v2 (cp -al). Les écritures
meta.json passent par tmp + os.replace → nouvel inode → l'original raw_v2
n'est JAMAIS modifié. Le déplacement d'un dossier ne touche pas non plus
l'archive d'origine.

Usage : python3 ihm/inspection_data.py        (aucun environnement ROS requis)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

# Palette + widgets partagés avec l'IHM principale (même dossier).
from ihm_vla import (BG, BG_PANEL, BG_CARD, FG, FG_DIM, ACCENT, ACCENT2,
                     ACCENT_BLUE, OK_GREEN, STOP_RED, WARN_YEL, LOG_BG,
                     ENTRY_BG, RButton, _lighten)

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE  = SCRIPT_DIR.parent
DATASETS   = WORKSPACE / "datasets"
ECHECS_DIR = DATASETS / "raw_echecs"


# ============================================================================
#  Accès données
# ============================================================================
def _list_raw_datasets() -> list[str]:
    """Datasets bruts inspectables : datasets/raw* contenant des episode_*/."""
    found = []
    if DATASETS.exists():
        for d in sorted(DATASETS.iterdir()):
            if d.is_dir() and d.name.startswith("raw") and any(d.glob("episode_*")):
                found.append(d.name)
    return found or ["raw_v2_1"]


def _read_meta(ep_dir: Path) -> dict:
    try:
        return json.loads((ep_dir / "meta.json").read_text())
    except Exception:  # noqa: BLE001
        return {}


def _write_meta(ep_dir: Path, meta: dict) -> None:
    """Écriture ATOMIQUE (tmp + replace) : casse le hardlink éventuel → le
    meta.json du dataset d'origine (ex. raw_v2) reste intact."""
    tmp = ep_dir / ".meta.tmp.json"
    tmp.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    os.replace(tmp, ep_dir / "meta.json")


# ============================================================================
#  Application
# ============================================================================
class InspectionData(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Inspection dataset — revue humaine des cycles")
        self.configure(bg=BG)
        self.geometry("1500x900")
        self.minsize(1180, 700)
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TCombobox", fieldbackground=ENTRY_BG, background=BG_PANEL,
                        foreground=FG, arrowcolor=FG)
        self.option_add("*TCombobox*Listbox.background", ENTRY_BG)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)

        # ── État ──
        self._dataset = tk.StringVar()
        self._eps: list[Path] = []       # dossiers épisodes du dataset courant
        self._idx = -1                   # épisode courant (index dans _eps)
        self._frames_front: list[Path] = []
        self._frames_wrist: list[Path] = []
        self._n_frames = 0
        self._cur = 0                    # frame courante
        self._playing = False
        # Mode de visionnage : "video" = lecture auto à l'ouverture d'un épisode ;
        # "frames" = pas de lecture, on se déplace aux flèches ←/→.
        self._view_mode = tk.StringVar(value="video")
        self._speed = tk.DoubleVar(value=1.0)
        self._acc = 0.0          # avance fractionnaire accumulée (voir _tick)
        self._fps = 15
        self._imgs: dict = {}            # refs PhotoImage (anti-GC)
        self._undo: list[tuple[Path, Path]] = []   # (destination, origine)
        self._seek_guard = False

        self._build_ui()
        self.bind("<space>", lambda _e: self._toggle_play())
        self.bind("<Left>",  lambda _e: self._step(-1))
        self.bind("<Right>", lambda _e: self._step(+1))
        self.bind("<Shift-Left>",  lambda _e: self._step(-10))
        self.bind("<Shift-Right>", lambda _e: self._step(+10))
        self.bind("<Up>",   lambda _e: self._select(self._idx - 1))
        self.bind("<Down>", lambda _e: self._select(self._idx + 1))
        self.bind("v", lambda _e: self._keep())
        self.bind("x", lambda _e: self._reject())
        self.bind("<Escape>", lambda _e: self.destroy())

        datasets = _list_raw_datasets()
        # Dataset demandé en argument (ex. `inspection_data.py raw_v2_exclus`),
        # sinon raw_v2_1 par défaut.
        asked = sys.argv[1] if len(sys.argv) > 1 else None
        if asked in datasets:
            self._dataset.set(asked)
        else:
            self._dataset.set("raw_v2_1" if "raw_v2_1" in datasets else datasets[0])
        self._load_dataset()
        self.focus_set()   # le clavier appartient au lecteur dès l'ouverture
        self.after(60, self._tick)

    # ------------------------------------------------------------------
    #  UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # -- Barre du haut : dataset + compteurs --
        top = tk.Frame(self, bg=BG_PANEL)
        top.pack(fill="x")
        tk.Label(top, text="Dataset :", bg=BG_PANEL, fg=FG,
                 font=("DejaVu Sans", 11, "bold")).pack(side="left", padx=(12, 6),
                                                        pady=8)
        self._combo = ttk.Combobox(top, textvariable=self._dataset,
                                   values=_list_raw_datasets(), state="readonly",
                                   width=18, takefocus=0,
                                   font=("DejaVu Sans Mono", 10))
        self._combo.pack(side="left")
        self._combo.bind("<<ComboboxSelected>>", lambda _e: self._load_dataset())
        # Sélecteur de mode : 🎬 Vidéo (lecture auto) / 🎞 Frames (flèches ←/→)
        modef = tk.Frame(top, bg=BG_PANEL)
        modef.pack(side="left", padx=(16, 0))
        tk.Label(modef, text="Mode :", bg=BG_PANEL, fg=FG,
                 font=("DejaVu Sans", 10)).pack(side="left", padx=(0, 4))
        for val, lbl in (("video", "🎬 Vidéo"), ("frames", "🎞 Frames")):
            tk.Radiobutton(modef, text=lbl, variable=self._view_mode, value=val,
                           command=self._on_mode_change, indicatoron=False,
                           bg=BG_CARD, fg=FG, selectcolor=ACCENT, takefocus=0,
                           activebackground=_lighten(BG_CARD, 0.1),
                           activeforeground=FG, bd=0, padx=10, pady=4,
                           font=("DejaVu Sans", 10, "bold")).pack(side="left",
                                                                  padx=1)
        self._counts_lbl = tk.Label(top, text="", bg=BG_PANEL, fg=FG_DIM,
                                    font=("DejaVu Sans", 10))
        self._counts_lbl.pack(side="left", padx=16)
        tk.Label(top, text="Espace ▶/⏸ · ←/→ frame · V garder · X rejeter",
                 bg=BG_PANEL, fg=FG_DIM, font=("DejaVu Sans", 9)).pack(
            side="right", padx=12)

        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True)

        # -- Liste des épisodes (gauche) --
        left = tk.Frame(main, bg=BG_PANEL)
        left.pack(side="left", fill="y", padx=(6, 0), pady=6)
        tk.Label(left, text="Épisodes", bg=BG_PANEL, fg=ACCENT,
                 font=("DejaVu Sans", 10, "bold")).pack(anchor="w", padx=8,
                                                        pady=(6, 2))
        lb_wrap = tk.Frame(left, bg=BG_PANEL)
        lb_wrap.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        # takefocus=0 : la liste ne capte JAMAIS le clavier → les flèches
        # restent dédiées au lecteur (souris pour choisir un épisode).
        self._lb = tk.Listbox(lb_wrap, width=30, bg=LOG_BG, fg=FG,
                              selectbackground=ACCENT, selectforeground="#15181c",
                              font=("DejaVu Sans Mono", 9), bd=0, takefocus=0,
                              highlightthickness=0, activestyle="none")
        sb = ttk.Scrollbar(lb_wrap, command=self._lb.yview)
        sb.pack(side="right", fill="y")
        self._lb.pack(side="left", fill="both", expand=True)
        self._lb.configure(yscrollcommand=sb.set)
        self._lb.bind("<<ListboxSelect>>", self._on_lb_select)

        # -- Zone centrale : vidéos + contrôles --
        center = tk.Frame(main, bg=BG)
        center.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        self._ep_lbl = tk.Label(center, text="—", bg=BG, fg=FG,
                                font=("DejaVu Sans", 13, "bold"))
        self._ep_lbl.pack(anchor="w", padx=4)
        self._meta_lbl = tk.Label(center, text="", bg=BG, fg=FG_DIM,
                                  font=("DejaVu Sans Mono", 10))
        self._meta_lbl.pack(anchor="w", padx=4)

        vids = tk.Frame(center, bg=LOG_BG)
        vids.pack(fill="both", expand=True, pady=6)
        vids.grid_rowconfigure(1, weight=1)
        self._vid_lbls: dict[str, tk.Label] = {}
        for i, name in enumerate(("front", "wrist")):
            vids.grid_columnconfigure(i, weight=1, uniform="v")
            tk.Label(vids, text=name.upper(), bg=LOG_BG, fg=FG_DIM,
                     font=("DejaVu Sans", 10, "bold")).grid(row=0, column=i,
                                                            pady=(6, 0))
            lbl = tk.Label(vids, bg=LOG_BG, fg=FG_DIM, text="—")
            lbl.grid(row=1, column=i, sticky="nsew", padx=6, pady=6)
            self._vid_lbls[name] = lbl

        # -- Timeline --
        tl = tk.Frame(center, bg=BG)
        tl.pack(fill="x")
        self._frame_lbl = tk.Label(tl, text="0 / 0", bg=BG, fg=FG_DIM, width=12,
                                   font=("DejaVu Sans Mono", 10))
        self._frame_lbl.pack(side="right", padx=6)
        self._scale = tk.Scale(tl, from_=0, to=1, orient="horizontal",
                               showvalue=False, command=self._on_seek,
                               bg=BG, fg=FG, troughcolor=ENTRY_BG, takefocus=0,
                               highlightthickness=0, activebackground=ACCENT,
                               sliderrelief="flat")
        self._scale.pack(fill="x", padx=6, expand=True)

        # -- Transport + verdicts --
        bar = tk.Frame(center, bg=BG)
        bar.pack(fill="x", pady=(4, 2))
        for txt, cmd in (("⏮", lambda: self._goto(0)),
                         ("−10", lambda: self._step(-10)),
                         ("◀", lambda: self._step(-1))):
            self._small_btn(bar, txt, cmd).pack(side="left", padx=2)
        self._play_btn = RButton(bar, "▶  Lecture", self._toggle_play,
                                 bg=ACCENT_BLUE, padx=18, pady=7)
        self._play_btn.pack(side="left", padx=6)
        for txt, cmd in (("▶", lambda: self._step(+1)),
                         ("+10", lambda: self._step(+10)),
                         ("⏭", lambda: self._goto(self._n_frames - 1))):
            self._small_btn(bar, txt, cmd).pack(side="left", padx=2)
        tk.Label(bar, text="Vitesse", bg=BG, fg=FG_DIM,
                 font=("DejaVu Sans", 9)).pack(side="left", padx=(14, 2))
        for s in (0.5, 1.0, 2.0, 4.0, 6.0, 8.0):
            tk.Radiobutton(bar, text=f"×{s:g}", variable=self._speed, value=s,
                           bg=BG, fg=FG_DIM, selectcolor=BG_PANEL,
                           activebackground=BG, takefocus=0,
                           font=("DejaVu Sans", 9)).pack(side="left")

        verd = tk.Frame(center, bg=BG)
        verd.pack(fill="x", pady=(2, 4))
        RButton(verd, "✓  GARDER  (V)", self._keep, bg=OK_GREEN,
                font=("DejaVu Sans", 11, "bold"), padx=22, pady=9).pack(
            side="left", padx=4)
        RButton(verd, "✗  REJETER → raw_echecs  (X)", self._reject, bg=STOP_RED,
                font=("DejaVu Sans", 11, "bold"), padx=22, pady=9).pack(
            side="left", padx=4)
        RButton(verd, "↩  Annuler dernier rejet", self._undo_reject, bg=BG_CARD,
                fg=FG, padx=14, pady=9).pack(side="left", padx=12)
        self._status = tk.Label(verd, text="", bg=BG, fg=FG_DIM,
                                font=("DejaVu Sans", 10), anchor="e")
        self._status.pack(side="right", padx=6, fill="x", expand=True)

    def _small_btn(self, parent: tk.Widget, txt: str, cmd) -> tk.Button:
        b = tk.Button(parent, text=txt, command=cmd, bg=BG_CARD, fg=FG, bd=0,
                      font=("DejaVu Sans", 10, "bold"), padx=10, pady=6,
                      activebackground=_lighten(BG_CARD, 0.15),
                      activeforeground=FG)
        b.bind("<Enter>", lambda _e: b.configure(bg=_lighten(BG_CARD, 0.12)),
               add="+")
        b.bind("<Leave>", lambda _e: b.configure(bg=BG_CARD), add="+")
        return b

    # ------------------------------------------------------------------
    #  Chargement dataset / épisode
    # ------------------------------------------------------------------
    def _load_dataset(self) -> None:
        self._playing = False
        root = DATASETS / self._dataset.get()
        self._eps = sorted(p for p in root.glob("episode_*") if p.is_dir())
        self._combo.configure(values=_list_raw_datasets())
        self._refresh_list()
        # Reprend au premier épisode PAS ENCORE inspecté.
        start = next((i for i, p in enumerate(self._eps)
                      if "human_check" not in _read_meta(p)), 0)
        self._select(start if self._eps else -1)

    def _ep_tag(self, ep: Path) -> tuple[str, str]:
        """(libellé listbox, couleur) selon meta : succès auto + verdict humain.
        Si l'épisode porte un `exclusion_code` (dataset raw_v2_exclus), il est
        affiché à la place du verdict — ces épisodes se REGARDENT, pas à trier."""
        m = _read_meta(ep)
        auto = "✓" if m.get("success") else "✗"
        code = m.get("exclusion_code")
        if code:
            return f"{ep.name[-6:]}  {auto}  {code}", WARN_YEL
        hc = m.get("human_check")
        if hc == "ok":
            return f"{ep.name[-6:]}  {auto}  [GARDÉ]", OK_GREEN
        if hc == "rejected":
            return f"{ep.name[-6:]}  {auto}  [REJETÉ]", STOP_RED
        return f"{ep.name[-6:]}  {auto}  [à voir]", FG

    def _refresh_list(self) -> None:
        self._lb.delete(0, "end")
        kept = rej = todo = 0
        for ep in self._eps:
            label, color = self._ep_tag(ep)
            self._lb.insert("end", label)
            self._lb.itemconfigure("end", foreground=color)
            if "[GARDÉ]" in label:
                kept += 1
            elif "[REJETÉ]" in label:
                rej += 1
            else:
                todo += 1
        self._counts_lbl.configure(
            text=f"{len(self._eps)} épisodes · ✓ gardés {kept} · "
                 f"✗ rejetés {rej} · restants {todo}")

    def _select(self, idx: int) -> None:
        if not self._eps or not (0 <= idx < len(self._eps)):
            return
        self._playing = False
        self._idx = idx
        ep = self._eps[idx]
        self._frames_front = sorted((ep / "obs_front").glob("frame_*.png"))
        self._frames_wrist = sorted((ep / "obs_wrist").glob("frame_*.png"))
        self._n_frames = min(len(self._frames_front), len(self._frames_wrist))
        m = _read_meta(ep)
        self._fps = int(m.get("fps", 15)) or 15
        auto = "✓ succès (auto)" if m.get("success") else "✗ échec (auto)"
        self._ep_lbl.configure(
            text=f"{ep.name}   ({idx + 1}/{len(self._eps)})   {auto}")
        reason = m.get("exclusion_reason", "")
        self._meta_lbl.configure(
            text=f"pick ({m.get('pick_x', '?')}, {m.get('pick_y', '?')}) m · "
                 f"{self._n_frames} frames @ {self._fps} fps"
                 + (f"\n⚠ EXCLU : {reason}" if reason else ""),
            fg=WARN_YEL if reason else FG_DIM)
        self._scale.configure(to=max(self._n_frames - 1, 1))
        self._lb.selection_clear(0, "end")
        self._lb.selection_set(idx)
        self._lb.see(idx)
        self._goto(0)
        # Mode 🎬 Vidéo : lecture auto à l'ouverture ; mode 🎞 Frames : on
        # reste sur la 1re frame et on navigue aux flèches ←/→.
        self._toggle_play(force=(self._view_mode.get() == "video"))

    def _on_mode_change(self) -> None:
        if self._view_mode.get() == "video":
            self._toggle_play(force=True)
        else:
            self._toggle_play(force=False)
        self.focus_set()

    def _on_lb_select(self, _e) -> None:
        sel = self._lb.curselection()
        if sel and sel[0] != self._idx:
            self._select(sel[0])
        self.focus_set()   # le clavier revient toujours au lecteur

    # ------------------------------------------------------------------
    #  Lecture
    # ------------------------------------------------------------------
    def _show_frame(self, i: int) -> None:
        if not self._n_frames:
            return
        i = max(0, min(i, self._n_frames - 1))
        self._cur = i
        for name, frames in (("front", self._frames_front),
                             ("wrist", self._frames_wrist)):
            lbl = self._vid_lbls[name]
            try:
                img = tk.PhotoImage(file=str(frames[i]))
                lw = lbl.winfo_width()
                if lw > 40 and img.width() > lw:
                    img = img.subsample(-(-img.width() // lw))
                self._imgs[name] = img
                lbl.configure(image=img, text="")
            except (tk.TclError, IndexError):
                pass
        self._frame_lbl.configure(text=f"{i + 1} / {self._n_frames}")
        self._seek_guard = True
        try:
            self._scale.set(i)
        finally:
            self._seek_guard = False

    _TICK_MS = 50   # 20 affichages/s max — le décodage de 2 PNG suit sans forcer

    def _tick(self) -> None:
        if self._playing and self._n_frames:
            # Avance fractionnaire accumulée : aux hautes vitesses (×4-×8) on
            # SAUTE des frames au lieu de toutes les décoder — sinon le
            # décodage PNG plafonne et la vitesse réelle stagne.
            self._acc += self._fps * self._speed.get() * self._TICK_MS / 1000.0
            step = int(self._acc)
            if step:
                self._acc -= step
                if self._cur >= self._n_frames - 1:
                    self._playing = False          # fin de cycle → pause
                    self._play_btn.set_text("▶  Lecture")
                else:
                    self._show_frame(min(self._cur + step, self._n_frames - 1))
        self.after(self._TICK_MS, self._tick)

    def _toggle_play(self, force: bool | None = None) -> None:
        if not self._n_frames:
            return
        self._playing = force if force is not None else not self._playing
        if self._playing and self._cur >= self._n_frames - 1:
            self._show_frame(0)                # relecture depuis le début
        self._play_btn.set_text("⏸  Pause" if self._playing else "▶  Lecture")

    def _step(self, d: int) -> None:
        self._playing = False
        self._play_btn.set_text("▶  Lecture")
        self._show_frame(self._cur + d)

    def _goto(self, i: int) -> None:
        self._show_frame(i)

    def _on_seek(self, value: str) -> None:
        i = int(float(value))
        # ⚠ Tk délivre le command de la Scale EN DIFFÉRÉ (boucle d'événements) :
        # le garde-fou de _show_frame est déjà retombé quand on arrive ici.
        # → on ne traite que les VRAIS déplacements utilisateur (valeur ≠ frame
        # courante) ; les échos des set() programmés sont ignorés, sinon chaque
        # avancée de lecture se mettait elle-même en pause.
        if self._seek_guard or i == self._cur:
            return
        self._playing = False
        self._play_btn.set_text("▶  Lecture")
        self._show_frame(i)

    # ------------------------------------------------------------------
    #  Verdicts
    # ------------------------------------------------------------------
    def _keep(self) -> None:
        if self._idx < 0:
            return
        ep = self._eps[self._idx]
        meta = _read_meta(ep)
        meta["human_check"] = "ok"
        meta["human_check_date"] = datetime.now().isoformat(timespec="seconds")
        _write_meta(ep, meta)
        self._status.configure(text=f"✓ {ep.name} gardé", fg=OK_GREEN)
        self._refresh_list()
        self._next_unchecked()

    def _reject(self) -> None:
        if self._idx < 0:
            return
        ep = self._eps[self._idx]
        ECHECS_DIR.mkdir(parents=True, exist_ok=True)
        dst = ECHECS_DIR / ep.name
        n = 1
        while dst.exists():                      # collision → suffixe
            dst = ECHECS_DIR / f"{ep.name}_rej{n}"
            n += 1
        meta = _read_meta(ep)
        meta["human_check"] = "rejected"
        meta["human_check_date"] = datetime.now().isoformat(timespec="seconds")
        meta["rejected_from"] = self._dataset.get()
        _write_meta(ep, meta)
        shutil.move(str(ep), str(dst))
        self._undo.append((dst, ep))
        try:
            dst_txt = dst.relative_to(WORKSPACE)
        except ValueError:           # dataset hors workspace → chemin complet
            dst_txt = dst
        self._status.configure(text=f"✗ {ep.name} → {dst_txt}", fg=STOP_RED)
        del self._eps[self._idx]
        self._refresh_list()
        self._idx = min(self._idx, len(self._eps) - 1)
        self._next_unchecked(start=self._idx)

    def _undo_reject(self) -> None:
        if not self._undo:
            self._status.configure(text="Rien à annuler.", fg=FG_DIM)
            return
        dst, orig = self._undo.pop()
        if not dst.exists():
            self._status.configure(text=f"Introuvable : {dst}", fg=WARN_YEL)
            return
        meta = _read_meta(dst)
        meta.pop("human_check", None)
        meta.pop("human_check_date", None)
        meta.pop("rejected_from", None)
        _write_meta(dst, meta)
        shutil.move(str(dst), str(orig))
        self._status.configure(text=f"↩ {orig.name} restauré", fg=ACCENT)
        self._load_dataset()

    def _next_unchecked(self, start: int | None = None) -> None:
        """Saute au prochain épisode sans verdict humain (boucle complète)."""
        if not self._eps:
            self._ep_lbl.configure(text="Dataset vide.")
            return
        s = self._idx + 1 if start is None else max(start, 0)
        order = list(range(s, len(self._eps))) + list(range(0, s))
        for i in order:
            if "human_check" not in _read_meta(self._eps[i]):
                self._select(i)
                return
        self._playing = False
        messagebox.showinfo(
            "Inspection terminée",
            "Tous les épisodes de ce dataset ont un verdict humain. ✓\n\n"
            "Pense à demander la reconversion LeRobot avant l'entraînement.")


# ============================================================================
def main() -> None:
    app = InspectionData()
    app.mainloop()


if __name__ == "__main__":
    main()
