"""Experimental front-bumper crossing metric for CARLA stop waypoints."""
import math


class SignalMonitor:
    def __init__(self, lines):
        self.lines = lines
        self.previous = None

    def update(self, front, states):
        events = []
        if self.previous is not None:
            previous, previous_states = self.previous
            for line in self.lines:
                x, y, yaw = line['pose']
                dx, dy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                before = (previous[0] - x) * dx + (previous[1] - y) * dy
                after = (front[0] - x) * dx + (front[1] - y) * dy
                if not before < 0 <= after:
                    continue
                fraction = -before / (after - before)
                cross = [a + fraction * (b - a) for a, b in zip(previous, front)]
                lateral = -(cross[0] - x) * dy + (cross[1] - y) * dx
                if abs(lateral) > line['width'] / 2 or abs(cross[2] - line['z']) > 2:
                    continue
                light_id = str(line['light_id'])
                prior, current = previous_states.get(light_id), states.get(light_id)
                if prior == current == 'Red':
                    kind = 'red_light'
                elif prior != current or current is None:
                    kind = 'signal_ambiguous'
                else:
                    continue
                events.append(dict(kind=kind, light_id=line['light_id'],
                                   stop_pose=line['pose'], previous_state=prior, state=current))
        self.previous = (list(front), dict(states))
        return events
