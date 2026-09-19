"""Measured three-lane Town05 traffic scenario, with a tracked slow lead car."""
import numpy as np


ROAD, START_S, EGO_LANE = 37, 200.0, -2
LANES = (-3, -2, -1)


def scenario_waypoint(world_map, lane=EGO_LANE, s=START_S):
    waypoint = world_map.get_waypoint_xodr(ROAD, lane, s)
    if waypoint is None or waypoint.is_junction or str(waypoint.lane_type) != 'Driving':
        raise RuntimeError('Expected non-junction driving lane in Town05 road 37')
    return waypoint


def spawn_transform(world_map):
    if world_map.name.split('/')[-1] != 'Town05':
        raise RuntimeError('Dense scenario requires Town05')
    initial = scenario_waypoint(world_map)
    lanes, todo = {initial.lane_id}, [initial]
    while todo:
        waypoint = todo.pop()
        for adjacent in (waypoint.get_left_lane(), waypoint.get_right_lane()):
            if (adjacent is not None and adjacent.road_id == ROAD and adjacent.lane_id < 0
                    and str(adjacent.lane_type) == 'Driving' and adjacent.lane_id not in lanes):
                lanes.add(adjacent.lane_id); todo.append(adjacent)
    if lanes != set(LANES):
        raise RuntimeError('Scenario must have exactly three same-direction lanes')
    for lane in LANES:
        for s in (START_S - 25, START_S, START_S + 180):
            scenario_waypoint(world_map, lane, s)
    transform = scenario_waypoint(world_map).transform
    transform.location.z += .3
    return transform


def populate(session):
    world_map = session.world.get_map()
    blueprints = session.world.get_blueprint_library()
    models = ['vehicle.lincoln.mkz_2017', 'vehicle.audi.a2', 'vehicle.toyota.prius', 'vehicle.tesla.model3']
    lead, specs = None, []
    for lane in LANES:
        offsets = [-22, -11, 14, 27, 40, 53, 66, 79] if lane == EGO_LANE else [-20, -9, 7, 20, 33, 46, 59, 72]
        for offset in offsets:
            wp = scenario_waypoint(world_map, lane, START_S + offset)
            transform = wp.transform
            transform.location.z += .3
            actor = session.world.try_spawn_actor(blueprints.find(models[len(specs) % len(models)]), transform)
            if actor is None:
                raise RuntimeError(f'Dense traffic spawn failed: lane {lane}, offset {offset}; refusing a sparse substitute')
            session.actors.append(actor); session.background_actors.append(actor)
            is_lead = lane == EGO_LANE and offset == 14
            speed = .8 if is_lead else (1.2 if lane == EGO_LANE else 2.0)
            actor.set_autopilot(True, session.tm_port)
            session.tm.auto_lane_change(actor, False)
            session.tm.distance_to_leading_vehicle(actor, 3.0)
            session.tm.set_desired_speed(actor, speed * 3.6)
            path = [scenario_waypoint(world_map, lane, s).transform.location
                    for s in np.arange(START_S + offset + 4, START_S + 190, 4)]
            session.tm.set_path(actor, path)
            specs.append(dict(id=actor.id, lane=lane, initial_s=wp.s, speed_mps=speed, lead=is_lead))
            if is_lead:
                lead = actor
    if lead is None or len(specs) != 24:
        raise RuntimeError('Incomplete dense scenario')
    session.metadata['dense_traffic'] = dict(road=ROAD, lanes=list(LANES), ego_lane=EGO_LANE,
                                           start_s=START_S, vehicles=specs, lead_id=lead.id)
    session.metadata['scene']['traffic_spawned'] = len(specs)
    return lead


class OvertakeMonitor:
    """Lead gap uses road longitudinal coordinate, not a changing ego heading."""
    def __init__(self):
        self.changed_lane = False
        self.passed_ticks = 0

    def update(self, ego_s, lead_s, lane, *, model_active, same_road, collision=False):
        gap = lead_s - ego_s
        if model_active and same_road and lane in LANES and lane != EGO_LANE:
            self.changed_lane = True
        passed = model_active and same_road and lane in LANES and self.changed_lane and gap < -8 and not collision
        self.passed_ticks = self.passed_ticks + 1 if passed else 0
        return dict(lead_gap_m=gap, passed_ticks=self.passed_ticks,
                    overtake='passed lead' if self.passed_ticks >= 10 else ('lane change observed' if self.changed_lane else 'not yet'))
