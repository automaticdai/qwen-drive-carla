import json

import numpy as np
from PIL import Image
import pytest

from qwen_drive_carla.route import RouteProgress
from qwen_drive_carla.run import run_episode
from qwen_drive_carla.session import CarlaSession


class FakeSession:
    def __init__(self, collision=False):
        self.ticks = 0
        self.controls = []
        self.autopilot_states = []
        self.collision = collision

    def tick(self):
        self.ticks += 1
        return dict(frame=self.ticks, timestamp=self.ticks / 10, pose=[self.ticks * .2, 0, 0],
                    velocity=[2, 0], acceleration=[0, 0], command="straight",
                    images={name: Image.new("RGB", (8, 8)) for name in ("front", "front_left", "front_right")})

    def apply(self, control):
        self.controls.append(control)

    def autopilot(self, enabled):
        self.autopilot_states.append(enabled)

    def drain_events(self):
        return [{"kind": "collision", "frame": self.ticks, "timestamp": self.ticks / 10}] if self.collision else []


class FakePlanner:
    def __init__(self, session, fail=False):
        self.session, self.fail, self.calls = session, fail, []

    def plan(self, history):
        from qwen_drive_carla.adapter import scene_payload
        payload = scene_payload(history, 15)
        assert payload["history"].shape == (16, 3)
        assert payload["views"]["<FRONT VIEW>"][-1].size == (8, 8)
        self.calls.append(self.session.ticks)
        if self.fail:
            raise RuntimeError("inference failed")
        return np.column_stack((np.arange(1, 51) * .2, np.zeros((50, 2)))), {"seconds": 1.0}


def route():
    return RouteProgress([[x, 0] for x in range(0, 102, 2)], ["follow"] * 51)


@pytest.mark.parametrize("mode", ["shadow", "closed-loop", "record"])
def test_complete_loop_modes(tmp_path, mode):
    session = FakeSession()
    planner = None if mode == "record" else FakePlanner(session)
    report = run_episode(session, planner, route(), tmp_path, mode=mode, steps=26)
    assert report["status"] == "time_limit" and report["ticks"] == 26
    assert session.controls[-1].brake == 1 and session.autopilot_states[-1] is False
    if planner:
        assert planner.calls == [16, 21, 26]
        assert report["plans"] == 3
    if mode == "closed-loop":
        assert all(c.brake == 1 for c in session.controls[:15])
        assert session.controls[15].reason == "tracking"
    else:
        assert len(session.controls) == 1  # final stop only
    records = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    assert len(records) == 26 and (tmp_path / records[0]["images"]["front"]).is_file()
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "time_limit"


def test_inference_failure_brakes_and_writes_report(tmp_path):
    session = FakeSession()
    with pytest.raises(RuntimeError, match="inference failed"):
        run_episode(session, FakePlanner(session, fail=True), route(), tmp_path, mode="closed-loop", steps=20)
    assert session.controls[-1].brake == 1
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "error"


def test_moving_handover_logs_only_post_handover_model_distance(tmp_path):
    session = FakeSession()
    result = run_episode(session, FakePlanner(session), route(), tmp_path, mode="closed-loop",
                         steps=21, warmup_driver="autopilot")
    assert result["handover_frame"] == 16
    assert result["distance_m"] == pytest.approx(4.0)
    assert result["model_distance_m"] == pytest.approx(1.0)
    assert len(session.controls) == 7  # six model controls plus final stop
    rows = [json.loads(l) for l in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert all(r["driver"] == "autopilot" and not r["control_applied"] for r in rows[:15])
    assert rows[15]["driver"] == "model" and rows[15]["control_applied"]


def test_collision_stops_before_inference(tmp_path):
    session = FakeSession(collision=True)
    planner = FakePlanner(session)
    report = run_episode(session, planner, route(), tmp_path, steps=20)
    assert report["status"] == "collision" and report["ticks"] == 1
    assert planner.calls == [] and session.controls[-1].brake == 1


def test_red_light_violation_stops_before_inference(tmp_path):
    session = FakeSession()
    session.drain_events = lambda: [dict(kind='red_light', frame=session.ticks)]
    planner = FakePlanner(session)
    report = run_episode(session, planner, route(), tmp_path, steps=20)
    assert report['status'] == 'red_light_violation'
    assert report['red_light_events'] == 1
    assert planner.calls == [] and session.controls[-1].brake == 1


@pytest.mark.parametrize("mode", ["shadow", "closed-loop"])
def test_rejected_predictions_are_saved_but_never_control_vehicle(tmp_path, mode):
    session = FakeSession()

    class InvalidPlanner:
        def plan(self, records):
            return np.full((50, 3), np.nan), {"seconds": 1.0}

    if mode == "shadow":
        result = run_episode(session, InvalidPlanner(), route(), tmp_path, mode=mode, steps=21)
        assert result["status"] == "time_limit" and result["rejected_plans"] == 2
        assert len(session.controls) == 1  # final stop, never model steering
    else:
        with pytest.raises(ValueError, match="finite"):
            run_episode(session, InvalidPlanner(), route(), tmp_path, mode=mode, steps=21)
        assert session.ticks == 16
    assert session.controls[-1].brake == 1
    logged = json.loads(next(tmp_path.glob("plan-*.json")).read_text())
    assert logged["validation_error"] and logged["trajectory"][0] == [None, None, None]


def test_session_cleanup_continues_after_actor_failure():
    calls = []

    class Actor:
        def destroy(self):
            calls.append("destroy")
            raise RuntimeError("gone")

        def stop(self):
            calls.append("stop")

    class World:
        def apply_settings(self, original):
            calls.append(original)

    session = CarlaSession()
    actor = Actor()
    session.actors, session.sensors = [actor], [actor]
    session.world, session.original = World(), "restored"
    with pytest.warns(UserWarning, match="cleanup"):
        session.close()
    assert calls == ["stop", "destroy", "restored"]
