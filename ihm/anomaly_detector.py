#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anomaly_detector.py — Détection d'anomalies par autoencoder LSTM
================================================================

Maintenance prédictive + sécurité IA :
  1. Collecte les profils articulaires (6 joints) sur 100 cycles nominaux.
  2. Entraîne un petit autoencoder LSTM (PyTorch) à les reconstruire.
  3. En production, calcule l'erreur de reconstruction de chaque cycle.
     - Profil normal → erreur faible.
     - Collision, surcharge, fatigue mécanique → erreur explose.
  4. Si l'erreur dépasse le seuil (moyenne + k·écart-type de l'apprentissage),
     le callback d'anomalie est déclenché → l'IHM arrête le mode.

Les imports de PyTorch/Numpy sont différés à l'usage : l'IHM peut donc démarrer
sur une RPi sans torch, le module reste inerte (statut « torch indisponible »)
mais ne fait pas planter l'interface.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Callable, Optional, Tuple


# === Hyperparamètres ========================================================
SEQ_LEN    = 50      # nombre de points par cycle après ré-échantillonnage
N_FEATURES = 6       # 6 articulations
N_NOMINAL  = 100     # profils nominaux à collecter avant entraînement
HIDDEN     = 16      # taille de l'état caché LSTM
N_EPOCHS   = 80
BATCH_SIZE = 8
LR         = 1e-3
THRESH_K   = 4.0     # seuil = moyenne + k·écart-type des erreurs d'entraînement
CYCLE_MIN_SAMPLES = 5
CYCLE_MIN_DUREE_S = 0.5
CYCLE_MAX_SAMPLES = 5000   # garde-fou si jamais aucun cycle ne se ferme


def _try_import_torch():
    """Importe torch + nn (différé). Retourne (None, None) si absent."""
    try:
        import torch
        from torch import nn
        return torch, nn
    except ImportError:
        return None, None


def _try_import_numpy():
    try:
        import numpy as np
        return np
    except ImportError:
        return None


# === Buffer d'un cycle ======================================================
class CycleBuffer:
    """Accumule (t, [6 angles]) puis ré-échantillonne à SEQ_LEN points."""

    def __init__(self):
        self._samples = []

    def add(self, angles_deg):
        self._samples.append((time.monotonic(), list(angles_deg)))
        if len(self._samples) > CYCLE_MAX_SAMPLES:
            self._samples = self._samples[-CYCLE_MAX_SAMPLES:]

    def reset(self):
        self._samples = []

    def __len__(self):
        return len(self._samples)

    def to_profile(self, seq_len=SEQ_LEN):
        """Renvoie un array (seq_len, 6) ou None si trop court."""
        if len(self._samples) < CYCLE_MIN_SAMPLES:
            return None
        np = _try_import_numpy()
        if np is None:
            return None
        ts   = np.array([s[0] for s in self._samples])
        angs = np.array([s[1] for s in self._samples])  # (N, 6)
        if ts[-1] - ts[0] < CYCLE_MIN_DUREE_S:
            return None
        ts_new = np.linspace(ts[0], ts[-1], seq_len)
        out = np.zeros((seq_len, N_FEATURES), dtype="float32")
        for j in range(N_FEATURES):
            out[:, j] = np.interp(ts_new, ts, angs[:, j])
        return out


# === Modèle : autoencoder LSTM ==============================================
def _build_model(torch, nn, seq_len=SEQ_LEN, n_features=N_FEATURES,
                 hidden=HIDDEN):
    class LSTMAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq_len = seq_len
            self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
            self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
            self.output  = nn.Linear(hidden, n_features)

        def forward(self, x):                  # x : (B, T, F)
            _, (h, _) = self.encoder(x)        # h : (1, B, H)
            z = h.squeeze(0).unsqueeze(1).repeat(1, self.seq_len, 1)
            y, _ = self.decoder(z)
            return self.output(y)

    return LSTMAutoencoder()


# === Détecteur principal ====================================================
class AnomalyDetector:
    """Collecte → Entraînement → Surveillance.

    Thread-safety : un seul verrou interne, les méthodes publiques sont sûres.
    L'entraînement est bloquant : l'IHM doit l'appeler depuis un Thread.
    """

    def __init__(self, base_dir: Path,
                 on_log: Optional[Callable[[str], None]] = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_path = self.base_dir / "profiles.npy"
        self.model_path    = self.base_dir / "model.pt"
        self.meta_path     = self.base_dir / "meta.json"
        self._log = on_log or (lambda s: None)

        self._lock = Lock()
        self._mode_actif = False
        self._cycle_buf  = CycleBuffer()
        self._profiles   = self._charger_profiles()
        self._meta       = self._charger_meta()
        self._model      = None
        self._torch      = None
        self._nn         = None

        self.surveillance_active  = False
        self.arret_auto           = True
        self._on_anomalie         = None
        self._derniere_erreur     = None

        if self.model_path.exists() and self._meta:
            self._charger_modele_si_possible()

    # ---------- persistance ------------------------------------------------
    def _charger_profiles(self):
        np = _try_import_numpy()
        if np is None or not self.profiles_path.exists():
            return None
        try:
            return np.load(self.profiles_path)
        except Exception:
            return None

    def _sauver_profiles(self):
        np = _try_import_numpy()
        if np is None or self._profiles is None:
            return
        try:
            np.save(self.profiles_path, self._profiles)
        except Exception as e:
            self._log("AnomalyDetector : sauvegarde profils KO (%s)" % e)

    def _charger_meta(self):
        if not self.meta_path.exists():
            return {}
        try:
            with open(self.meta_path) as f:
                return json.load(f)
        except Exception:
            return {}

    def _sauver_meta(self):
        try:
            with open(self.meta_path, "w") as f:
                json.dump(self._meta, f, indent=2)
        except Exception as e:
            self._log("AnomalyDetector : sauvegarde meta KO (%s)" % e)

    def _charger_modele_si_possible(self):
        torch, nn = _try_import_torch()
        if torch is None:
            return False
        self._torch, self._nn = torch, nn
        try:
            model = _build_model(
                torch, nn,
                seq_len   = int(self._meta.get("seq_len", SEQ_LEN)),
                n_features= int(self._meta.get("n_features", N_FEATURES)),
                hidden    = int(self._meta.get("hidden", HIDDEN)),
            )
            model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            model.eval()
            self._model = model
            return True
        except Exception as e:
            self._log("AnomalyDetector : modèle non chargé (%s)" % e)
            return False

    # ---------- statut -----------------------------------------------------
    def torch_disponible(self):
        torch, _ = _try_import_torch()
        return torch is not None

    def nb_profils(self):
        with self._lock:
            if self._profiles is None:
                return 0
            return int(self._profiles.shape[0])

    def est_entraine(self):
        return self._model is not None

    def statut(self):
        if not self.torch_disponible():
            return "torch_absent"
        if self.est_entraine():
            return "entraine"
        if self.nb_profils() >= N_NOMINAL:
            return "pret_a_entrainer"
        return "collecte"

    def derniere_erreur(self):
        return self._derniere_erreur, self._meta.get("threshold")

    def meta(self):
        return dict(self._meta)

    # ---------- callbacks --------------------------------------------------
    def set_callback_anomalie(self, cb: Callable[[float, float], None]):
        self._on_anomalie = cb

    # ---------- cycle de vie d'un mode ------------------------------------
    def mode_demarre(self):
        with self._lock:
            self._mode_actif = True
            self._cycle_buf.reset()

    def mode_arrete(self):
        with self._lock:
            self._mode_actif = False
            self._cycle_buf.reset()

    def enregistrer_echantillon(self, angles_deg):
        if not self._mode_actif or angles_deg is None:
            return
        if len(angles_deg) != N_FEATURES:
            return
        self._cycle_buf.add(angles_deg)

    def cycle_termine(self) -> Optional[Tuple[str, Optional[float]]]:
        """À appeler quand le mode signale un cycle terminé.

        Retourne :
          ("trop_court", None)  : cycle inexploitable
          ("collect", n)        : profil ajouté, n profils collectés
          ("ok", err)           : surveillance OK
          ("anomalie", err)     : surveillance KO, callback déclenché
          None                  : torch indisponible
        """
        with self._lock:
            profil = self._cycle_buf.to_profile()
            self._cycle_buf.reset()
        if profil is None:
            return ("trop_court", None)
        if not self.est_entraine():
            self._ajouter_profil(profil)
            return ("collect", float(self.nb_profils()))
        return self._evaluer(profil)

    def _ajouter_profil(self, profil):
        np = _try_import_numpy()
        if np is None:
            return
        with self._lock:
            if self._profiles is None:
                self._profiles = profil[None, :, :].copy()
            else:
                self._profiles = np.concatenate(
                    [self._profiles, profil[None, :, :]], axis=0)
            if self._profiles.shape[0] > N_NOMINAL * 2:
                self._profiles = self._profiles[-N_NOMINAL * 2:]
            self._sauver_profiles()

    # ---------- entraînement ----------------------------------------------
    def entrainer(self,
                  on_progress: Optional[Callable[[int, int, float], None]] = None
                  ) -> Tuple[bool, str]:
        """Bloquant. À lancer dans un Thread."""
        np = _try_import_numpy()
        torch, nn = _try_import_torch()
        if np is None:
            return False, "Numpy n'est pas installé (pip install numpy)."
        if torch is None:
            return False, "PyTorch n'est pas installé (pip install torch)."
        with self._lock:
            if self._profiles is None or self._profiles.shape[0] < N_NOMINAL:
                return False, ("Pas assez de profils nominaux (%d/%d)."
                               % (self.nb_profils(), N_NOMINAL))
            data = self._profiles.copy().astype("float32")

        # Normalisation z-score par joint
        mean = data.mean(axis=(0, 1))
        std  = data.std(axis=(0, 1)) + 1e-6
        norm = (data - mean) / std

        device = "cpu"
        x = torch.tensor(norm, dtype=torch.float32, device=device)

        model = _build_model(torch, nn).to(device)
        opt    = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        model.train()
        n = x.size(0)
        for epoch in range(N_EPOCHS):
            perm = torch.randperm(n)
            losses = []
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                xb = x[idx]
                recon = model(xb)
                loss = loss_fn(recon, xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            moy = sum(losses) / max(1, len(losses))
            if on_progress is not None:
                try:
                    on_progress(epoch + 1, N_EPOCHS, moy)
                except Exception:
                    pass

        # Seuil : moyenne + k·écart-type des erreurs sur le train set
        model.eval()
        with torch.no_grad():
            recon = model(x)
            errs = ((recon - x) ** 2).mean(dim=(1, 2)).cpu().numpy()
        err_mean = float(errs.mean())
        err_std  = float(errs.std())
        threshold = err_mean + THRESH_K * err_std

        # Sauvegarde
        try:
            torch.save(model.state_dict(), self.model_path)
        except Exception as e:
            return False, "Sauvegarde modèle KO : %s" % e

        self._meta = {
            "seq_len":     SEQ_LEN,
            "n_features":  N_FEATURES,
            "hidden":      HIDDEN,
            "mean":        mean.tolist(),
            "std":         std.tolist(),
            "err_mean":    err_mean,
            "err_std":     err_std,
            "threshold":   threshold,
            "thresh_k":    THRESH_K,
            "n_profiles":  int(data.shape[0]),
            "trained_at":  datetime.now().isoformat(timespec="seconds"),
        }
        self._sauver_meta()

        self._torch, self._nn = torch, nn
        self._model = model
        return True, ("Entraînement terminé sur %d profils. "
                      "Seuil = %.5f (μ=%.5f, σ=%.5f)."
                      % (int(data.shape[0]), threshold, err_mean, err_std))

    # ---------- inférence --------------------------------------------------
    def _evaluer(self, profil):
        np = _try_import_numpy()
        torch = self._torch
        if np is None or torch is None or self._model is None:
            return None
        mean = np.array(self._meta["mean"], dtype="float32")
        std  = np.array(self._meta["std"], dtype="float32")
        norm = (profil - mean) / std
        x = torch.tensor(norm[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            recon = self._model(x)
            err = float(((recon - x) ** 2).mean().item())
        self._derniere_erreur = err
        thresh = float(self._meta.get("threshold", 0.0))
        if err > thresh and self.surveillance_active:
            if self.arret_auto and self._on_anomalie is not None:
                try:
                    self._on_anomalie(err, thresh)
                except Exception as e:
                    self._log("AnomalyDetector : callback KO (%s)" % e)
            return ("anomalie", err)
        return ("ok", err)

    # ---------- reset ------------------------------------------------------
    def reset(self, supprimer_profils=True):
        """Repart à zéro. Si supprimer_profils=False, on garde le dataset."""
        with self._lock:
            self._model = None
            self._meta  = {}
            self._derniere_erreur = None
            self.surveillance_active = False
            try:
                if self.model_path.exists():
                    self.model_path.unlink()
                if self.meta_path.exists():
                    self.meta_path.unlink()
                if supprimer_profils:
                    self._profiles = None
                    if self.profiles_path.exists():
                        self.profiles_path.unlink()
            except Exception as e:
                self._log("AnomalyDetector : reset partiel (%s)" % e)
            self._cycle_buf.reset()
