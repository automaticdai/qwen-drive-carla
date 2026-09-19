import base64
import gzip
import json

import numpy as np
from PIL import Image
import pytest

from qwen_drive_carla.image_transport import DebugImageEncoder, ServerImageCache, ImageCacheMiss, unpack_body
from qwen_drive_carla.adapter import scene_payload, input_snapshot
from qwen_drive_carla.bev import SURROUND
from qwen_drive_carla.remote import decode_scene
from qwen_drive_carla.debug_inference import decode_surround


def sample():
    rng=np.random.default_rng(7)
    image=Image.fromarray(rng.integers(0,256,(192,320,3),dtype=np.uint8)).resize((640,384))
    images={name:image for name in ('front','front_left','front_right')}
    images.update({name:image.resize((896,512)) for name,_,_ in SURROUND})
    records=[dict(frame=i+1,timestamp=i*.1,pose=[0,i,90],velocity=[0,2],acceleration=[1,0],command='straight',images=images) for i in range(16)]
    cameras={name:dict(fov=90,mount_matrix=np.eye(4).tolist()) for name,_,_ in SURROUND}
    return records,cameras


def test_cached_lossless_roundtrip_and_pixel_equality():
    records,cameras=sample();payload=scene_payload(records,15)
    encoder=DebugImageEncoder();cache=ServerImageCache();sizes=[(320,192)]*3+[(640,384)]
    body,metrics=encoder.pack(payload,records,cameras,sizes,'test')
    request=json.loads(unpack_body(body,'gzip'));scene,surround=cache.resolve(request)
    decoded=decode_scene(scene);views,_=decode_surround(surround)
    expected=records[0]['images']['front'].resize((320,192),Image.Resampling.BICUBIC)
    np.testing.assert_array_equal(decoded['views']['<FRONT VIEW>'][0],expected)
    np.testing.assert_array_equal(views['CAM_BACK'],records[-1]['images']['CAM_BACK'])
    next_body,next_metrics=encoder.pack(payload,records,cameras,sizes,'next',cache.images)
    assert next_metrics['images_uploaded']==0 and next_metrics['images_reused']==18
    assert next_metrics['encoder_cache_hits']==18 and len(next_body)<len(body)/10
    cache.resolve(json.loads(unpack_body(next_body,'gzip')))


def test_jpeg_is_explicit_and_valid():
    records,cameras=sample();payload=scene_payload(records,15)
    body,metrics=DebugImageEncoder('jpeg').pack(payload,records,cameras,[(320,192)]*3+[(640,384)],'jpeg')
    assert metrics['image_codec']=='jpeg-quality95-444'
    scene,_=ServerImageCache().resolve(json.loads(unpack_body(body,'gzip')))
    assert decode_scene(scene)['views']['<FRONT VIEW>'][-1].size==(640,384)


def test_cache_eviction_and_corruption_do_not_invent_images():
    cache=ServerImageCache(max_items=0)
    records,cameras=sample();encoder=DebugImageEncoder()
    body,_=encoder.pack(scene_payload(records,15),records,cameras,[(320,192)]*3+[(640,384)],'one')
    req=json.loads(unpack_body(body,'gzip'));cache.resolve(req)
    assert not cache.images
    req['image_uploads']={}
    with pytest.raises(ImageCacheMiss):cache.resolve(req)
    req['image_uploads']={'bad-hash':base64.b64encode(b'image').decode()}
    with pytest.raises(ValueError,match='digest'):cache.resolve(req)


def test_compressed_body_must_be_complete():
    body=gzip.compress(b'{}')
    assert unpack_body(body,'gzip')==b'{}'
    with pytest.raises(ValueError):unpack_body(body[:-2],'gzip')
    with pytest.raises(ValueError):unpack_body(body+body,'gzip')


def test_compressed_body_expansion_is_bounded(monkeypatch):
    import qwen_drive_carla.image_transport as transport
    monkeypatch.setattr(transport,'LIMIT',128)
    with pytest.raises(ValueError,match='oversized'):
        transport.unpack_body(gzip.compress(b'x'*129),'gzip')


def test_cache_miss_retry_does_not_run_inference_twice(monkeypatch):
    import threading
    import qwen_drive_carla.remote as remote
    from qwen_drive_carla.debug_inference import RemoteDebugPlanner
    monkeypatch.setattr(remote,'ServerImageCache',lambda:ServerImageCache(max_items=0))
    class Planner:
        calls=0
        loading_info=dict(bev=True,debug_image_cache=True,image_sizes=[(320,192)]*3+[(640,384)])
        def debug_payload(self,payload,surround):
            self.calls+=1
            return np.ones((50,3)),dict(seconds=.01),dict(source_frame=payload['token'])
    planner=Planner();server=remote.make_server(planner,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        client=RemoteDebugPlanner(f'http://127.0.0.1:{server.server_port}',transport='jpeg')
        records,cameras=sample()
        client.debug(records,cameras)
        client.known_images={digest for digest,_ in client.encoder.cache.values()}
        _,response=client.debug(records,cameras)
        assert response['metrics']['cache_resend']
        assert planner.calls==2  # The rejected reference-only request never inferred.
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5)


def test_instrument_snapshot_matches_ego_input_axes():
    records,_=sample();payload=scene_payload(records,15)
    snapshot=input_snapshot(payload,records[-1]['timestamp'])
    np.testing.assert_allclose(snapshot['ego_velocity'],[2,0],atol=1e-12)
    np.testing.assert_allclose(snapshot['ego_acceleration'],[0,1],atol=1e-12)
    assert snapshot['timestamp']==1.5 and len(snapshot['history_velocity'])==16
