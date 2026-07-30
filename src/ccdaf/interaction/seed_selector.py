"""
SeedSelector
============

Thin composition wrapper that binds three independent layers:

* :class:`SeedStateMachine` — pure logic (queue, history, undo).
* :class:`SeedGeometryResolver` — pure geometry (snapping, validation).
* A PyVista interaction surface — pick callbacks, markers, HUD.

This module contains NO business logic. Every decision about
"is this pick acceptable?" is delegated:

    xyz ──► geometry.snap()        (deterministic vertex id)
        ──► geometry.validate_pv() (anatomical prior, PV names only)
        ──► state.try_commit()     (order + duplicate enforcement)

The public surface is intentionally unchanged from the previous
implementation: ``main_app.py`` and the rest of the pipeline continue
to call ``start``, ``stop``, ``undo_last``, ``reset``, ``seeds``,
``is_complete``, ``seeds_for_tagging``, and so on.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pyvista as pv


from ccdaf.core.seed_state_machine import (
    SEED_ORDER,
    Seed,
    SeedStateMachine,
    CommitResult,
)
from ccdaf.core.seed_geometry import SeedGeometryResolver, GeometryError
from ccdaf.core.seed_profiles import (
    SeedProfile,
    SEED_PROFILE,
    SEED_PROMPT,
    SEED_COLOR,
)


class SeedSelector:
    """Sequential seed picker built on SeedStateMachine + SeedGeometryResolver.

    The point set to pick — its order, prompts, colours, pulmonary-vein
    prior and export key — comes from a :class:`SeedProfile`. The default
    profile reproduces the original six-seed workflow, so callers that do
    not pass one see no change.
    """

    def __init__(
        self,
        mesh: pv.PolyData,
        plotter,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_complete: Optional[Callable[[Dict[str, Seed]], None]] = None,
        profile: SeedProfile = SEED_PROFILE,
    ) -> None:
        self.mesh = mesh
        self.plotter = plotter
        self.on_progress = on_progress
        self.on_complete = on_complete
        self._profile = profile

        self._state = SeedStateMachine(profile.order)
        self._geom = SeedGeometryResolver(mesh)

        self._active: bool = False
        self._marker_actors: Dict[str, object] = {}
        self._label_actor = None
        self._hud_actor = None
        self._warning_actor = None

    # ------------------------------------------------------------------
    # Public API (unchanged surface)
    # ------------------------------------------------------------------
    def _enable_picking(self) -> None:
        """(Re)install the point-picking callback on the plotter.

        ``picker="hardware"`` selects VTK's z-buffer picker: it returns the
        front-most *visible* surface point under the cursor, so a pick can
        never bleed through to an occluded back-wall vertex (the old
        ``vtkPointPicker`` default snapped to the nearest vertex to the pick
        *ray*, including hidden ones). ``pickable_window=False`` means the
        callback fires only on a real surface hit — clicks on empty space are
        dropped by PyVista before reaching :meth:`_on_pick`.
        """
        self.plotter.enable_point_picking(
            callback=self._on_pick,
            picker="hardware",
            use_picker=True,
            show_message=False,
            show_point=False,
            pickable_window=False,
            left_clicking=True,
        )

    def start(self) -> None:
        """Begin seed selection from a clean slate — wipes any existing seeds."""
        if self._active:
            return
        self._active = True
        self._state.reset()
        self._clear_markers()
        self._enable_picking()
        self._refresh_hud()
        self._emit_progress()

    def resume(self) -> None:
        """Re-enable picking without touching existing seed state.

        Intended for the "undo after completion" flow: when the last
        seed is removed we want the user to pick the missing one back,
        not lose all the earlier seeds.
        """
        if self._active:
            return
        self._active = True
        self._enable_picking()
        self._refresh_hud()
        self._emit_progress()

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        try:
            self.plotter.disable_picking()
        except Exception:
            pass
        self._clear_warning()

    def hide(self) -> None:
        """Remove this selector's actors from the plotter, keeping seed state.

        Used when another seed set becomes the active one: the picks are
        retained (and re-exportable), only their markers/labels/HUD leave
        the view so the two sets do not clutter each other.
        """
        self.stop()
        self._clear_markers()  # markers + labels + warning
        try:
            self.plotter.remove_actor("seed_hud", reset_camera=False)
        except Exception:
            pass
        self._hud_actor = None

    def show(self) -> None:
        """Re-draw markers, labels and HUD from the retained seed state.

        The inverse of :meth:`hide`; does not re-enable picking (call
        :meth:`resume` for that).
        """
        for seed in self._state.seeds.values():
            self._add_marker(seed)
        self._refresh_hud()

    def undo_last(self) -> Optional[str]:
        removed = self._state.undo()
        if removed is not None:
            self._remove_marker(removed)
            self._emit_progress()
        return removed

    def reset(self) -> None:
        self._state.reset()
        self._clear_markers()
        self._emit_progress()

    # ------------------------------------------------------------------
    @property
    def seeds(self) -> Dict[str, Seed]:
        return self._state.seeds

    @property
    def is_complete(self) -> bool:
        return self._state.is_complete

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def state(self) -> SeedStateMachine:
        """Escape hatch for headless tests (pure-logic layer)."""
        return self._state

    @property
    def profile(self) -> SeedProfile:
        return self._profile

    def next_name(self) -> Optional[str]:
        return self._state.next_name()

    def next_prompt(self) -> str:
        nxt = self._state.next_name()
        return ("All seeds collected." if nxt is None
                else self._profile.prompts[nxt])

    def seeds_for_tagging(self) -> Dict[str, int]:
        return {n: s.vertex_id for n, s in self._state.seeds.items()
                if n not in self._profile.no_tag_names}

    def apply_positions(self, positions) -> "list[str]":
        """Rebuild the selection from saved coordinates, in order.

        Each coordinate is snapped to the current surface by nearest
        point — a saved seed carries no vertex id worth trusting across a
        clip or a refinement — and then walks the same gauntlet as a
        click: duplicate spacing, PV prior, state-machine order. Stops at
        the first seed that is missing or rejected, leaving the earlier
        ones placed and the selection resumable from there.

        Returns one message per seed that could not be placed; empty
        means the selection completed and ``on_complete`` fired.
        """
        problems: list = []
        self.stop()
        self._state.reset()
        self._clear_markers()
        for name in self._profile.order:
            if name not in positions:
                problems.append(f"{name}: not in the file")
                break
            try:
                snap = self._geom.snap(np.asarray(positions[name], dtype=float))
            except GeometryError as exc:
                problems.append(f"{name}: {exc}")
                break
            existing_xyz = [s.xyz for s in self._state.seeds.values()]
            if self._geom.is_duplicate_position(snap.xyz, existing_xyz):
                problems.append(f"{name}: lands on an already-placed seed")
                break
            if name in self._profile.pv_names:
                ok, reason = self._geom.validate_pv(snap.vertex_id)
                if not ok:
                    problems.append(f"{name}: {reason}")
                    break
            res = self._state.try_commit(
                Seed(name=name, vertex_id=snap.vertex_id, xyz=snap.xyz),
            )
            if res is not CommitResult.OK:
                problems.append(f"{name}: {res.value}")
                break
            self._add_marker(self._state.seeds[name])
        self._emit_progress()
        if self._state.is_complete and self.on_complete is not None:
            self.on_complete(self._state.seeds)
        return problems

    # ------------------------------------------------------------------
    # Pick routing
    # ------------------------------------------------------------------
    def _on_pick(self, picked_point, *args, **kwargs) -> None:
        if not self._active or picked_point is None:
            return
        if self._state.is_complete:
            return

        # ``picked_point`` is already the front-most *visible* surface hit
        # (hardware/z-buffer picker), so no camera ray-trace correction is
        # needed — snap it straight to the nearest mesh vertex.
        try:
            snap = self._geom.snap(picked_point)
        except GeometryError as exc:
            self._warn(f"Pick rejected: {exc}")
            return

        name = self._state.next_name()
        if name is None:
            return

        existing_xyz = [s.xyz for s in self._state.seeds.values()]
        if self._geom.is_duplicate_position(snap.xyz, existing_xyz):
            self._warn(
                "Ignored: that point is too close to an existing seed."
            )
            return

        if name in self._profile.pv_names:
            ok, reason = self._geom.validate_pv(snap.vertex_id)
            if not ok:
                self._warn(f"{name}: {reason}")
                return

        res = self._state.try_commit(
            Seed(name=name, vertex_id=snap.vertex_id, xyz=snap.xyz),
        )
        if res is not CommitResult.OK:
            self._warn(f"Pick rejected by state machine: {res.value}")
            return

        seed = self._state.seeds[name]
        self._add_marker(seed)
        self._clear_warning()
        self._emit_progress()

        if self._state.is_complete:
            self._active = False
            try:
                self.plotter.disable_picking()
            except Exception:
                pass
            if self.on_complete is not None:
                self.on_complete(self._state.seeds)

    # ------------------------------------------------------------------
    # HUD / warnings
    # ------------------------------------------------------------------
    def _refresh_hud(self) -> None:
        try:
            self.plotter.remove_actor("seed_hud", reset_camera=False)
        except Exception:
            pass
        self._hud_actor = None

        nxt = self._state.next_name()
        if nxt is None:
            text, color = "All seeds collected.", "white"
        else:
            text = (f"[{len(self._state.seeds)}/{len(self._profile.order)}]  "
                    f"NEXT: {nxt} — {self._profile.prompts[nxt]}")
            color = self._profile.colors.get(nxt, "white")
        try:
            self._hud_actor = self.plotter.add_text(
                text, position="lower_left", color=color,
                font_size=11, name="seed_hud",
            )
        except Exception:
            pass

    def _warn(self, msg: str) -> None:
        self._clear_warning()
        try:
            self._warning_actor = self.plotter.add_text(
                msg, position="upper_edge", color="yellow",
                font_size=11, name="seed_warning",
            )
            self.plotter.render()
        except Exception:
            pass

    def _clear_warning(self) -> None:
        try:
            self.plotter.remove_actor("seed_warning", reset_camera=False)
        except Exception:
            pass
        self._warning_actor = None

    # ------------------------------------------------------------------
    # Markers
    # ------------------------------------------------------------------
    def _marker_radius(self) -> float:
        return max(self._geom.diag * 0.01, 1e-3)

    def _add_marker(self, seed: Seed) -> None:
        sphere = pv.Sphere(
            radius=self._marker_radius(),
            center=np.asarray(seed.xyz, dtype=float).tolist(),
        )
        actor = self.plotter.add_mesh(
            sphere,
            color=self._profile.colors[seed.name],
            name=f"seed_{seed.name}",
            reset_camera=False,
            pickable=False,
        )
        self._marker_actors[seed.name] = actor
        self._refresh_labels()

    def _remove_marker(self, name: str) -> None:
        actor = self._marker_actors.pop(name, None)
        if actor is not None:
            try:
                self.plotter.remove_actor(actor, reset_camera=False)
            except Exception:
                pass
        self._refresh_labels()

    def _clear_markers(self) -> None:
        for name in list(self._marker_actors.keys()):
            self._remove_marker(name)
        self._marker_actors.clear()
        if self._label_actor is not None:
            try:
                self.plotter.remove_actor(self._label_actor, reset_camera=False)
            except Exception:
                pass
            self._label_actor = None
        self._clear_warning()

    def _refresh_labels(self) -> None:
        if self._label_actor is not None:
            try:
                self.plotter.remove_actor(self._label_actor, reset_camera=False)
            except Exception:
                pass
            self._label_actor = None
        if not self._state.seeds:
            return
        pts = np.array([np.asarray(s.xyz, dtype=float)
                        for s in self._state.seeds.values()])
        names = [s.name for s in self._state.seeds.values()]
        self._label_actor = self.plotter.add_point_labels(
            pts, names,
            point_size=1, font_size=14, text_color="white",
            shape_opacity=0.4, reset_camera=False,
            name="seed_labels", always_visible=True,
        )

    # ------------------------------------------------------------------
    def _emit_progress(self) -> None:
        self._refresh_hud()
        if self.on_progress is not None:
            self.on_progress(
                self._state.next_name() or "",
                len(self._state.seeds),
                len(self._profile.order),
            )


__all__ = [
    "SeedSelector",
    "Seed",
    "SEED_ORDER",
    "SEED_COLOR",
    "SEED_PROMPT",
]
