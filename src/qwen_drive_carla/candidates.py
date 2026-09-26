"""Bounded Qwen trajectory candidates and explicit CARLA-assisted selection."""
from dataclasses import asdict, dataclass

import numpy as np

from .adapter import ego_vectors
from .controller import TrajectoryTracker
from .safety import Decision

MAX_CANDIDATES = 6


def candidate_count(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= value <= MAX_CANDIDATES:
        raise ValueError(f'Candidate count must be an integer from 1 to {MAX_CANDIDATES}')
    return int(value)


def candidate_array(value, *, finite=False):
    points = np.asarray(value, dtype=float)
    if points.ndim != 3 or points.shape[1:] != (50, 3):
        raise ValueError('Expected candidates with shape (N, 50, 3)')
    candidate_count(len(points))
    if finite and not np.isfinite(points).all():
        raise ValueError('Non-finite trajectory candidates')
    return points


@dataclass
class Selection:
    index: int | None
    decision: Decision
    evaluations: list

    def log(self):
        return dict(selected_index=self.index, count=len(self.evaluations),
                    valid_count=sum(e['decision']['reason'] == 'clear' for e in self.evaluations),
                    candidates=self.evaluations,
                    score_formula='progress_m - 0.25*acceleration_rms - heading_variation_rad - 0.1*lateral_displacement_m')


def select_candidate(candidates, guard, pose, speed, obstacles=(), *, progress=None, max_speed=4.):
    """Safety is a gate, never a weighted tradeoff against progress.

    Progress is signed distance along the scenario road if a callback is supplied;
    otherwise ego-forward displacement. Scores are heuristic, not probabilities.
    The actual tick control is checked again by the guard after selection.
    """
    candidates = candidate_array(candidates)
    evaluations, decisions = [], []
    for index, points in enumerate(candidates):
        decision = guard.check_plan(points, pose, obstacles)
        metrics, score = {}, None
        if decision.safe:
            tracker = TrajectoryTracker(max_speed=max_speed, wheelbase=guard.wheelbase,
                                        max_steer_degrees=np.degrees(guard.max_steer))
            tracker.set_plan(points, pose, 0.)
            proposed = tracker.step(pose, speed, 0.)
            metrics.update(initial_target_speed=proposed.target_speed, initial_control_reason=proposed.reason)
            decision = guard.check_path(*guard.stopping_path(pose, speed, proposed), obstacles)
        if decision.safe:
            world_end = ego_vectors(points[-1:, :2], pose[2])[0] + np.asarray(pose[:2])
            distance = progress(*world_end)-progress(*pose[:2]) if progress else float(points[-1, 0])
            if not np.isfinite(distance):
                decision = Decision('invalid_progress')
            else:
                # Cap reward at the speed limit; implausibly fast plans get no extra reward.
                distance = float(np.clip(distance, -max_speed*5., max_speed*5.))
                positions = np.vstack((np.zeros(2), points[:, :2]))
                speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1)/.1
                # 0.3-second average reduces finite-difference quantization noise in scoring only.
                smoothed = np.convolve(speeds, np.ones(3)/3., mode='valid')
                acceleration_rms = float(np.sqrt(np.mean((np.diff(smoothed)/.1)**2)))
                heading_variation = float(np.abs(np.diff(np.unwrap(np.r_[0., points[:, 2]]))).sum())
                lateral = float(abs(points[-1, 1]))
                metrics.update(progress_m=distance, acceleration_rms=acceleration_rms,
                               heading_variation_rad=heading_variation, lateral_displacement_m=lateral)
                score = distance - .25*acceleration_rms - heading_variation - .1*lateral
        decisions.append(decision)
        evaluations.append(dict(index=index, decision=asdict(decision), score=score, **metrics))
    valid = [e for e in evaluations if e['decision']['reason'] == 'clear']
    # Stable ties preserve the first sample, making one-candidate comparisons straightforward.
    best = max(valid, key=lambda e: e['score']) if valid else None
    index = best['index'] if best else None
    return Selection(index, decisions[index] if index is not None else decisions[0], evaluations)
