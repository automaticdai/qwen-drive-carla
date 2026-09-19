"""Own simulator ticks, sensors and actors for a single experiment."""
import queue
import random
import time
import warnings

import numpy as np
from PIL import Image
from .signals import SignalMonitor

def image_for_frame(messages, frame, timeout):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for camera frame {frame}")
        try:
            image = messages.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError(f"Timed out waiting for camera frame {frame}") from exc
        if image.frame == frame:
            return image
        if image.frame > frame:
            raise RuntimeError(f"Camera skipped frame {frame}; received {image.frame}")



class CarlaSession:
    def __init__(self, host="127.0.0.1", port=2000, tm_port=8000, spawn_index=0, seed=42, timeout=30, camera_profile="small", camera_rig="planning", spawn_transform=None):
        if camera_profile not in ("small", "high"):
            raise ValueError("camera_profile must be small or high")
        self.camera_size = (640, 384) if camera_profile == "small" else (1280, 768)
        if camera_rig not in ('planning', 'surround', 'debug'):
            raise ValueError('Unknown camera rig')
        self.camera_rig = camera_rig
        self.spawn_transform = spawn_transform
        if camera_rig == 'surround':
            self.camera_size = (896, 512)
        self.host, self.port, self.tm_port = host, port, tm_port
        self.spawn_index, self.seed, self.timeout = spawn_index, seed, timeout
        self.actors, self.sensors, self.queues = [], [], {}
        self.events = queue.SimpleQueue()
        self.world = self.original = self.tm = None
        self.original_weather = None
        self.follow_camera = False
        self.background_actors = []
        self.metadata = {"fixed_delta_seconds": 0.1, "seed": seed, "cameras": {},
                         "vehicle": "vehicle.tesla.model3", "coordinate_frame": "CARLA world; yaw degrees"}

    def __enter__(self):
        import carla
        self.carla = carla
        client = carla.Client(self.host, self.port)
        client.set_timeout(self.timeout)
        if client.get_client_version() != client.get_server_version():
            raise RuntimeError("CARLA client and server versions must match")
        self.world = client.get_world()
        original = self.world.get_settings()
        if original.synchronous_mode:
            raise RuntimeError("Use a dedicated asynchronous server; another client may own the ticks")
        self.original = original
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            settings.substepping = True
            settings.max_substep_delta_time = 0.01
            settings.max_substeps = 10
            settings.no_rendering_mode = False
            self.world.apply_settings(settings)
            self.tm = client.get_trafficmanager(self.tm_port)
            self.tm.set_synchronous_mode(True)
            self.tm.set_random_device_seed(self.seed)
            world_map = self.world.get_map()
            self.metadata.update(carla_version=client.get_server_version(), map=world_map.name)
            blueprints = self.world.get_blueprint_library()
            spawns = world_map.get_spawn_points()
            if not 0 <= self.spawn_index < len(spawns):
                raise ValueError(f"spawn-index must be in [0, {len(spawns) - 1}]")
            self.vehicle = self.world.spawn_actor(blueprints.find("vehicle.tesla.model3"), self.spawn_transform or spawns[self.spawn_index])
            self.actors.append(self.vehicle)
            self.signal_actors = list(self.world.get_actors().filter('traffic.traffic_light'))
            stop_lines = []
            for light in self.signal_actors:
                for waypoint in light.get_stop_waypoints():
                    t = waypoint.transform
                    stop_lines.append(dict(light_id=light.id, width=waypoint.lane_width,
                                           pose=[t.location.x, t.location.y, t.rotation.yaw], z=t.location.z))
            self.signal_monitor = SignalMonitor(stop_lines)
            self.metadata['signal_metric'] = dict(version=1, reference='front bumper midpoint', stop_lines=stop_lines)
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
            from .bev import SURROUND
            camera_mounts = [(name, yaw) for name, _, yaw in SURROUND] if self.camera_rig == 'surround' else [("front", 0), ("front_left", -60), ("front_right", 60)]
            if self.camera_rig == 'debug':
                camera_mounts += [(name, yaw) for name, _, yaw in SURROUND]
            for name, yaw in camera_mounts:
                bp = blueprints.find("sensor.camera.rgb")
                is_surround = name.startswith('CAM_')
                width, height = (896, 512) if is_surround else self.camera_size
                for key, value in {"image_size_x": str(width), "image_size_y": str(height), "fov": "90", "sensor_tick": "0.0", "enable_postprocess_effects": "true"}.items():
                    bp.set_attribute(key, value)
                mount = carla.Transform(carla.Location(x=0 if is_surround else 1.5, z=1.7), carla.Rotation(yaw=yaw))
                sensor = self.world.spawn_actor(bp, mount, attach_to=self.vehicle)
                self.actors.append(sensor)
                self.sensors.append(sensor)
                messages = queue.Queue()
                sensor.listen(messages.put)
                self.queues[name] = messages
                self.metadata["cameras"][name] = {"width": width, "height": height, "fov": 90, "x": 1.5, "z": 1.7, "yaw": yaw}
                self.metadata['cameras'][name].update(x=mount.location.x, mount_matrix=mount.get_matrix())
            for name in ("collision", "lane_invasion"):
                sensor = self.world.spawn_actor(blueprints.find(f"sensor.other.{name}"), carla.Transform(), attach_to=self.vehicle)
                self.actors.append(sensor)
                self.sensors.append(sensor)
                sensor.listen(lambda event, kind=name: self.record_event(kind, event))
            return self
        except BaseException:
            self.close()
            raise

    def autopilot(self, enabled):
        self.vehicle.set_autopilot(enabled, self.tm_port)
        if enabled:
            self.tm.auto_lane_change(self.vehicle, False)

    def record_event(self, kind, event):
        data = dict(kind=kind, frame=event.frame, timestamp=event.timestamp)
        if kind == "lane_invasion":
            data["markings"] = [str(marking.type) for marking in event.crossed_lane_markings]
        else:
            data["other_actor"] = dict(id=event.other_actor.id, type=event.other_actor.type_id)
        self.events.put(data)

    def configure_scene(self, *, weather="ClearNoon", traffic=0, follow_camera=False):
        """Seeded background traffic; all actors and weather belong to this session."""
        self.follow_camera = follow_camera
        self.original_weather = self.world.get_weather()
        self.world.set_weather(getattr(self.carla.WeatherParameters, weather))
        rng = random.Random(self.seed)
        spawns = self.world.get_map().get_spawn_points()
        origin = spawns[self.spawn_index].location
        candidates = [s for s in spawns if 12 < s.location.distance(origin) < 120]
        rng.shuffle(candidates)
        blueprints = sorted((b for b in self.world.get_blueprint_library().filter("vehicle.*")
                             if b.has_attribute("number_of_wheels") and
                             b.get_attribute("number_of_wheels").as_int() == 4), key=lambda b: b.id)
        spawned = []
        for transform in candidates:
            if len(spawned) >= traffic:
                break
            blueprint = rng.choice(blueprints)
            actor = self.world.try_spawn_actor(blueprint, transform)
            if actor is None:
                continue
            self.actors.append(actor)
            self.background_actors.append(actor)
            actor.set_autopilot(True, self.tm_port)
            self.tm.auto_lane_change(actor, False)
            self.tm.set_desired_speed(actor, 18.0)
            spawned.append(dict(id=actor.id, blueprint=blueprint.id,
                                x=transform.location.x, y=transform.location.y,
                                yaw=transform.rotation.yaw))
        self.metadata["scene"] = dict(weather=weather, traffic_requested=traffic,
                                      traffic_spawned=len(spawned), actors=spawned)

    def apply(self, control):
        self.vehicle.apply_control(self.carla.VehicleControl(throttle=control.throttle, brake=control.brake, steer=control.steer))

    def tick(self, command="straight"):
        frame = self.world.tick(self.timeout)
        snapshot = self.world.get_snapshot()
        if snapshot.frame != frame:
            raise RuntimeError("Unexpected world tick from another client")
        state = snapshot.find(self.vehicle.id)
        if state is None:
            raise RuntimeError("Ego vehicle is missing from the snapshot")
        transform = state.get_transform()
        if self.follow_camera:
            location = transform.location - transform.get_forward_vector() * 8 + self.carla.Location(z=4)
            self.world.get_spectator().set_transform(self.carla.Transform(
                location, self.carla.Rotation(pitch=-18, yaw=transform.rotation.yaw)))
        velocity, acceleration = state.get_velocity(), state.get_acceleration()
        images = {}
        for name, messages in self.queues.items():
            data = image_for_frame(messages, frame, self.timeout)
            if abs(data.timestamp - snapshot.timestamp.elapsed_seconds) > 1e-4:
                raise RuntimeError("Camera and ego timestamps disagree")
            rgb = np.frombuffer(data.raw_data, dtype=np.uint8).reshape(data.height, data.width, 4)
            images[name] = Image.fromarray(rgb[:, :, [2, 1, 0]])
        distances = [transform.location.distance(actor_state.get_transform().location)
                     for actor in self.background_actors
                     if (actor_state := snapshot.find(actor.id)) is not None]
        light = self.vehicle.get_traffic_light() if self.vehicle.is_at_traffic_light() else None
        signal_states = {str(actor.id): str(actor.get_state()) for actor in self.signal_actors}
        box = self.vehicle.bounding_box
        front = transform.transform(self.carla.Location(
            x=box.location.x + box.extent.x, y=box.location.y, z=box.location.z))
        for event in self.signal_monitor.update([front.x, front.y, front.z], signal_states):
            self.events.put(dict(event, frame=frame, timestamp=snapshot.timestamp.elapsed_seconds))
        return dict(frame=frame, timestamp=snapshot.timestamp.elapsed_seconds,
                    nearest_traffic_m=min(distances) if distances else None,
                    at_traffic_light=self.vehicle.is_at_traffic_light(),
                    traffic_light_id=light.id if light else None,
                    traffic_light_state=str(light.get_state()) if light else None,
                    signal_states=signal_states, signal_check_available=bool(self.signal_monitor.lines),
                    pose=[transform.location.x, transform.location.y, transform.rotation.yaw],
                    velocity=[velocity.x, velocity.y], acceleration=[acceleration.x, acceleration.y], command=command, images=images)

    def drain_events(self):
        result = []
        while True:
            try:
                result.append(self.events.get_nowait())
            except queue.Empty:
                return result

    def close(self):
        errors = []
        # Unregister background vehicles before destroying them: Traffic Manager
        # otherwise retains handles and can abort during interpreter shutdown.
        for actor in self.background_actors:
            try:
                actor.set_autopilot(False, self.tm_port)
            except RuntimeError as exc:
                errors.append(str(exc))
        self.background_actors.clear()
        for sensor in self.sensors:
            try:
                sensor.stop()
            except RuntimeError as exc:
                errors.append(str(exc))
        for actor in reversed(self.actors):
            try:
                actor.destroy()
            except RuntimeError as exc:
                errors.append(str(exc))
        self.sensors.clear()
        self.actors.clear()
        try:
            if self.tm is not None:
                self.tm.set_synchronous_mode(False)
        except RuntimeError as exc:
            errors.append(str(exc))
        finally:
            if self.original_weather is not None:
                try:
                    self.world.set_weather(self.original_weather)
                except RuntimeError as exc:
                    errors.append(str(exc))
                self.original_weather = None
            if self.original is not None:
                try:
                    self.world.apply_settings(self.original)
                except RuntimeError as exc:
                    errors.append(str(exc))
                self.original = None
        self.metadata["cleanup_errors"] = errors
        if errors:
            warnings.warn("CARLA cleanup reported errors: " + "; ".join(errors))

    def __exit__(self, *exc):
        self.close()


def save_record(record, directory, handle):
    import json
    images = {}
    for name, image in record["images"].items():
        relative = f"{name}/{record['frame']:08d}.png"
        (directory / name).mkdir(exist_ok=True)
        image.save(directory / relative)
        images[name] = relative
    handle.write(json.dumps({**record, "images": images}) + "\n")
    handle.flush()
