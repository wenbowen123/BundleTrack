#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import zmq
import cv2
from tqdm import tqdm


def empty_network(feed_dict):
    empty_output = {
        "kpts": np.zeros((0, 2), dtype=np.float32),
        "feats": np.zeros((0, 128), dtype=np.float32),
    }
    return empty_output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/r2d2.yml")
    parser.add_argument("--ckpt", type=str, default="checkpoints/r2d2/r2d2.ckpt")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = f"tcp://*:{args.port}"
    print("port", port)
    socket.bind(port)

    while 1:
        print(f"FNN listending to {port}")
        msgs = socket.recv_multipart(0)
        assert len(msgs) == 2, "#msgs={}".format(len(msgs))
        wh = np.frombuffer(msgs[0], dtype=np.int32)
        W = wh[0]
        H = wh[1]
        print(f"W={W}, H={H}")
        msg = msgs[1]
        photo = np.frombuffer(msg, dtype=np.uint8).reshape(H, W, -1).squeeze()
        photo_ori = photo.copy()

        rgb = photo.copy()
        if photo.ndim == 3 and photo.shape[-1] == 3:
            photo = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
        photo = photo[None, ..., None].astype(np.float32) / 255.0
        assert photo.ndim == 4

        feed_dict = {
            "photo_ph": photo,
        }

        outs = empty_network(feed_dict=feed_dict)

        num_feat = len(outs["kpts"])
        feat_dim = outs["feats"].shape[1]
        msg = np.array([num_feat, feat_dim]).reshape(-1).astype(np.int32).tobytes()
        socket.send(msg, 2)
        msg = outs["kpts"].astype(np.float32).reshape(-1).tobytes()
        socket.send(msg, 2)
        msg = outs["feats"].astype(np.float32).reshape(-1).tobytes()
        socket.send(msg, 0)
