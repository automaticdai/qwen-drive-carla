import json
import threading
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import pytest

from qwen_drive_carla.bev import SURROUND
from qwen_drive_carla.dashboard import Dashboard
from qwen_drive_carla.debug_inference import RemoteDebugPlanner, decode_surround, image_string
from qwen_drive_carla.dense_traffic import OvertakeMonitor
from qwen_drive_carla.remote import make_server


def surround():
    return dict(images={n:image_string(Image.new('RGB',(896,512))) for n,_,_ in SURROUND},
                cameras={n:dict(fov=90,mount_matrix=np.eye(4).tolist()) for n,_,_ in SURROUND})


def test_surround_requires_all_views_and_rigid_calibration():
    data=surround()
    images,calibration=decode_surround(data)
    assert len(images)==len(calibration)==6
    data['cameras']['CAM_BACK']['mount_matrix'][0][0]=2
    with pytest.raises(ValueError,match='rigid'):
        decode_surround(data)
    del data['images']['CAM_BACK']
    with pytest.raises(ValueError,match='six'):
        decode_surround(data)


def test_shared_debug_request_frame_matching():
    class Planner:
        loading_info=dict(bev=True)
        def debug_payload(self,payload,bev):
            assert len(bev[0])==6
            return np.ones((50,3)),dict(seconds=.1),dict(source_frame=payload['token'])
    server=make_server(Planner(),0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        client=RemoteDebugPlanner(f'http://127.0.0.1:{server.server_port}')
        cameras=surround()['cameras']
        images={k:Image.new('RGB',(640,384)) for k in ('front','front_left','front_right')}
        images.update({n:Image.new('RGB',(896,512)) for n,_,_ in SURROUND})
        records=[dict(frame=i+1,timestamp=i*.1,pose=[i,0,0],velocity=[1,0],acceleration=[0,0],command='straight',images=images) for i in range(16)]
        points,response=client.debug(records,cameras)
        assert points.shape==(50,3) and response['bev']['source_frame']=='16'
        assert response['qwen_inputs']['source_frame']=='16'
        assert response['qwen_inputs']['ego_velocity']==[1.,0.]
        client.request=lambda *args:dict(protocol=1,request_id='stale',token='15')
        with pytest.raises(RuntimeError,match='requested frame'):
            client.debug(records,cameras)
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5)


def test_overtake_requires_model_lane_change_and_sustained_pass():
    m=OvertakeMonitor()
    for _ in range(12):
        assert m.update(20,0,-2,model_active=True,same_road=True)['overtake']=='not yet'
    for _ in range(9):
        assert m.update(20,0,-1,model_active=True,same_road=True)['overtake']!='passed lead'
    assert m.update(20,0,-1,model_active=True,same_road=True)['overtake']=='passed lead'
    assert m.update(20,0,-1,model_active=True,same_road=True,collision=True)['passed_ticks']==0
    assert m.update(20,0,1,model_active=True,same_road=False)['passed_ticks']==0


def test_dashboard_pause_step_stop_and_cross_origin_rejection():
    dashboard=Dashboard(0)
    try:
        dashboard.control('pause');dashboard.control('step')
        assert dashboard.allow_tick() and dashboard.paused and dashboard.steps==0
        dashboard.control('stop');assert not dashboard.allow_tick()
        base=f'http://127.0.0.1:{dashboard.server.server_port}'
        with urlopen(base+'/state') as r:
            assert json.load(r)['stop_requested']
        request=Request(base+'/control',data=b'{"action":"run"}',headers={'Content-Type':'application/json','Origin':'https://other.invalid'})
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as exc:
            urlopen(request)
        assert exc.value.code==403
    finally:
        dashboard.close()
