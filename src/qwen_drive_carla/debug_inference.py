"""Shared-VLM sequential planning and BEV inference for live debugging."""
import argparse
import base64
from io import BytesIO
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image

from .adapter import scene_payload, input_snapshot
from .bev import SURROUND, camera_calibration
from .remote import RemotePlanner, VERSION, MAX_REQUEST, encode_scene, json_bytes, make_server
from .image_transport import DebugImageEncoder, ImageCacheMiss


def image_string(image, format='PNG'):
    stream = BytesIO()
    image.save(stream, format=format)
    return base64.b64encode(stream.getvalue()).decode('ascii')


def decode_surround(data):
    names = [name for name, _, _ in SURROUND]
    if not isinstance(data, dict) or set(data) != {'images', 'cameras'}:
        raise ValueError('Expected surround images and calibration')
    if set(data['images']) != set(names) or set(data['cameras']) != set(names):
        raise ValueError('Expected six surround cameras')
    images, calibration = {}, []
    for name in names:
        raw = base64.b64decode(data['images'][name], validate=True)
        with Image.open(BytesIO(raw)) as image:
            if image.format not in ('PNG', 'WEBP', 'JPEG') or image.size != (896, 512):
                raise ValueError('Expected 896x512 PNG/WebP/JPEG surround image')
            images[name] = image.convert('RGB')
        spec = data['cameras'][name]
        k, r, t = camera_calibration(spec['mount_matrix'], 896, 512, spec['fov'])
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r), 1, atol=1e-4):
            raise ValueError('Camera transform is not rigid')
        calibration.append((k, r, t))
    return images, calibration


class RemoteDebugPlanner(RemotePlanner):
    def __init__(self, url, timeout=120, transport='lossless'):
        super().__init__(url, timeout)
        if transport not in ('lossless', 'legacy', 'jpeg'):
            raise ValueError('Unknown debug transport')
        self.transport = transport
        self.encoder = DebugImageEncoder('jpeg' if transport == 'jpeg' else 'webp')
        self.known_images = set()

    def debug(self, records, cameras):
        started = time.perf_counter()
        payload = scene_payload(records, len(records) - 1)
        request_id = uuid.uuid4().hex
        current = records[-1]
        optimized = self.transport != 'legacy' and self.loading_info.get('debug_image_cache')
        stages = {}
        if optimized:
            body, stages = self.encoder.pack(payload, records, cameras, self.loading_info['image_sizes'], request_id, self.known_images)
            wire_started = time.perf_counter()
            try:
                response = self.request('/debug', body, 'gzip')
            except ImageCacheMiss:
                # The server rejects missing references before inference. One full
                # resend is safe; timeouts/model errors are never retried.
                self.known_images.clear()
                body, retry_stages = self.encoder.pack(payload, records, cameras, self.loading_info['image_sizes'], request_id)
                response = self.request('/debug', body, 'gzip')
                stages.update(cache_resend=True, retry_encode_seconds=retry_stages['encode_seconds'])
                stages['wire_bytes'] += retry_stages['wire_bytes']
                stages['images_uploaded'] += retry_stages['images_uploaded']
                stages['images_reused'] = retry_stages['images_reused']
            stages['http_seconds'] = time.perf_counter() - wire_started
            self.known_images = set(response.get('cached_image_ids', []))
        else:
            surround = dict(images={n: image_string(current['images'][n]) for n, _, _ in SURROUND},
                            cameras={n: cameras[n] for n, _, _ in SURROUND})
            body = json_bytes(dict(protocol=VERSION, request_id=request_id,
                                   scene=encode_scene(payload, self.loading_info.get('image_sizes')), surround=surround))
            if len(body) > MAX_REQUEST:
                raise ValueError('Debug request exceeds transport limit')
            stages.update(encode_seconds=time.perf_counter()-started, image_codec='png', wire_bytes=len(body))
            wire_started = time.perf_counter()
            response = self.request('/debug', body)
            stages['http_seconds'] = time.perf_counter()-wire_started
        if response.get('protocol') != VERSION or response.get('request_id') != request_id or response.get('token') != payload['token']:
            raise RuntimeError('Debug response does not match requested frame')
        trajectory = np.asarray(response['trajectory'], dtype=float)
        if trajectory.shape != (50, 3) or not np.isfinite(trajectory).all():
            raise ValueError('Invalid debug trajectory')
        response['metrics'].update(request_seconds=time.perf_counter() - started,
                                   request_bytes=stages.get('wire_bytes', len(body)))
        response['metrics'].update(stages)
        response['qwen_inputs'] = input_snapshot(payload, current['timestamp'])
        return trajectory, response


def shared_planner(model_path, image_profile):
    from .planner import QwenPlanner
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionProcessor, PerceptionFrame
    from qwen_drive_perception.configuration_perception import MAP_PALETTE, OCC_PALETTE
    from transformers import AutoTokenizer

    class DebugPlanner(QwenPlanner):
        def __init__(self):
            super().__init__(model_path, image_profile=image_profile)
            self.perception = QwenDrivePerception.from_pretrained(str(Path(model_path) / 'perception'),
                dtype=self.torch.bfloat16, local_files_only=True).to('cuda').eval()
            self.processor = PerceptionProcessor(AutoTokenizer.from_pretrained(str(model_path), local_files_only=True))
            self.perception.attach(self.model.vlm, self.processor)
            self.loading_info.update(bev=True, shared_vlm=True, bev_image_size=[896, 512], debug_image_cache=True)

        def debug_payload(self, payload, surround):
            images, calibration = surround

            class MemoryFrame(PerceptionFrame):
                def __init__(self):
                    self.token, self.dataset_type = payload['token'], 'nuscenes'
                    self.cam_order = [n for n, _, _ in SURROUND]
                    self.content = []
                    for name, tag, _ in SURROUND:
                        self.content.extend([dict(text='<' + tag + '>'), dict(image=name)])
                    self.content.append(dict(text='Analyze the scene.'))
                    self.cam_intrinsic = np.stack([x[0] for x in calibration])
                    self.sensor2lidar_rotation = np.stack([x[1] for x in calibration])
                    self.sensor2lidar_translation = np.stack([x[2] for x in calibration])
                    self.lidar2ego = np.eye(4)
                def image(self, camera):
                    return images[camera]

            trajectory, metrics = self.plan_payload(payload)
            # Release planning workspaces before the different BEV allocation pattern.
            self.torch.cuda.empty_cache()
            self.torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with self.torch.inference_mode():
                inputs, metadata = self.processor(MemoryFrame(), device='cuda')
                result = self.perception.infer(inputs, metadata)
            self.torch.cuda.synchronize()
            metrics.update(bev_seconds=time.perf_counter() - started,
                           bev_peak_reserved_bytes=self.torch.cuda.max_memory_reserved())
            result = {k: v.detach().float().cpu().numpy() if self.torch.is_tensor(v) else v for k, v in result.items()}
            if any(not np.isfinite(v).all() for v in result.values()):
                raise ValueError('Non-finite BEV prediction')
            occ = result['occ'].astype(int)
            z = np.where(occ != 9, np.arange(occ.shape[2]), -1).max(axis=2)
            top = np.take_along_axis(occ, np.maximum(z, 0)[..., None], axis=2)[..., 0]
            top[z < 0] = 9
            map_rgb = np.asarray(MAP_PALETTE, dtype=np.uint8)[result['map'].astype(int)].transpose(1, 0, 2)[::-1, ::-1]
            occ_rgb = np.asarray(OCC_PALETTE, dtype=np.uint8)[top][::-1, ::-1]
            bev = dict(boxes=result['boxes'].tolist(), scores=result['scores'].tolist(), labels=result['labels'].tolist(),
                       map_png=image_string(Image.fromarray(map_rgb)), occ_png=image_string(Image.fromarray(occ_rgb)),
                       source_frame=payload['token'], feeds_planner=False, out_of_distribution=True,
                       map_extent=[-15, 15, -30, 30], occupancy_extent=[-40, 40, -40, 40])
            del inputs, result
            self.torch.cuda.empty_cache()
            return trajectory, metrics, bev

    return DebugPlanner()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/Qwen-Drive-1.0-4B'))
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--image-profile', choices=['small', 'high'], default='high')
    args = parser.parse_args()
    with make_server(shared_planner(args.model, args.image_profile), args.port) as server:
        print('Shared planning + BEV service ready', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
