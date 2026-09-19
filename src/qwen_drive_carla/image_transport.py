"""Bounded image reuse and lossless WebP transport for debug requests."""
import base64
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
from io import BytesIO
import json
import threading
import time
import zlib

import numpy as np
from PIL import Image

from .adapter import VIEWS
from .bev import SURROUND

LIMIT = 64 * 1024 * 1024


class ImageCacheMiss(RuntimeError):
    pass


def unpack_body(body, encoding):
    if encoding in (None, '', 'identity'):
        return body
    if encoding != 'gzip':
        raise ValueError('Unsupported content encoding')
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    result = decoder.decompress(body, LIMIT + 1)
    if len(result) > LIMIT or not decoder.eof or decoder.unused_data:
        raise ValueError('Invalid or oversized compressed request')
    return result


class ServerImageCache:
    def __init__(self, max_bytes=LIMIT, max_items=160):
        self.images = OrderedDict()
        self.bytes = 0
        self.max_bytes, self.max_items = max_bytes, max_items

    def resolve(self, request):
        uploads = request.get('image_uploads', {})
        if not isinstance(uploads, dict) or len(uploads) > 18:
            raise ValueError('Invalid image upload table')
        decoded = {}
        for digest, encoded in uploads.items():
            raw = base64.b64decode(encoded, validate=True)
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError('Image digest mismatch')
            decoded[digest] = encoded
        used = set()
        def resolve_ref(ref):
            if not isinstance(ref, dict) or set(ref) != {'image_id'} or not isinstance(ref['image_id'], str):
                raise ValueError('Invalid image reference')
            digest = ref['image_id']
            value = decoded.get(digest, self.images.get(digest))
            if value is None:
                raise ImageCacheMiss('Image cache entry expired')
            used.add(digest)
            return value
        scene = dict(request['scene'])
        scene['views'] = {name:[resolve_ref(ref) for ref in refs] for name,refs in scene['views'].items()}
        surround = dict(request['surround'])
        surround['images'] = {name:resolve_ref(ref) for name,ref in surround['images'].items()}
        for digest in used:
            value = decoded.get(digest, self.images.get(digest))
            old = self.images.pop(digest, None)
            if old is not None:
                self.bytes -= len(old)
            self.images[digest] = value
            self.bytes += len(value)
        while self.bytes > self.max_bytes or len(self.images) > self.max_items:
            _, value = self.images.popitem(last=False)
            self.bytes -= len(value)
        return scene, surround


class DebugImageEncoder:
    def __init__(self, codec='webp', max_items=96):
        if codec not in ('webp', 'jpeg'):
            raise ValueError('Unknown image codec')
        self.codec, self.max_items = codec, max_items
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def encode(self, image, frame, name, size):
        key = (frame, name, tuple(size), self.codec)
        with self.lock:
            if key in self.cache:
                result = self.cache.pop(key); self.cache[key] = result
                return result, True
        if image.size != tuple(size):
            image = image.resize(tuple(size), Image.Resampling.BICUBIC)
        stream = BytesIO()
        if self.codec == 'webp':
            image.save(stream, format='WEBP', lossless=True, method=0)
        else:
            image.save(stream, format='JPEG', quality=95, subsampling=0)
        raw = stream.getvalue()
        result = (hashlib.sha256(raw).hexdigest(), base64.b64encode(raw).decode('ascii'))
        with self.lock:
            self.cache[key] = result
            while len(self.cache) > self.max_items:
                self.cache.popitem(last=False)
        return result, False

    def pack(self, payload, records, cameras, image_sizes, request_id, known=()):
        started = time.perf_counter()
        numeric = ('history','history_velocity','history_acceleration','ego_velocity','ego_acceleration')
        scene = {k:np.asarray(payload[k]).tolist() for k in numeric}
        scene.update(nav_command=payload['nav_command'],driving_command=payload['driving_command'],token=payload['token'])
        scene['views'] = {tag:[] for tag in VIEWS.values()}
        surround = dict(images={}, cameras={name:cameras[name] for name,_,_ in SURROUND})
        jobs = []
        for name, tag in VIEWS.items():
            for index, history_index in enumerate((0,5,10,15)):
                record = records[history_index]
                jobs.append((record['images'][name],record['frame'],name,tuple(image_sizes[index])))
        current = records[-1]
        jobs.extend((current['images'][name],current['frame'],name,(896,512)) for name,_,_ in SURROUND)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda job:self.encode(*job),jobs))
        uploads, reused = {}, 0
        known = set(known)
        for index, ((digest,encoded),_) in enumerate(results):
            ref = dict(image_id=digest)
            if index < 12:
                scene['views'][list(VIEWS.values())[index//4]].append(ref)
            else:
                surround['images'][SURROUND[index-12][0]] = ref
            if digest not in known:
                uploads[digest] = encoded
            else:
                reused += 1
        encoded_at = time.perf_counter()
        request = dict(protocol=1,request_id=request_id,scene=scene,surround=surround,image_uploads=uploads)
        raw = json.dumps(request,allow_nan=False,separators=(',',':')).encode()
        if len(raw)>LIMIT:
            raise ValueError('Debug request exceeds decoded size limit')
        body = gzip.compress(raw,compresslevel=1)
        return body,dict(encode_seconds=encoded_at-started,serialize_seconds=time.perf_counter()-encoded_at,
                         encoder_cache_hits=sum(hit for _,hit in results),images_reused=reused,
                         images_uploaded=len(uploads),json_bytes=len(raw),wire_bytes=len(body),
                         image_codec='lossless-webp' if self.codec=='webp' else 'jpeg-quality95-444')
