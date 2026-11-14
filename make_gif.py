"""
Render a clip of the game straight from MuJoCo (no viewer window, no screen recorder) and
save it as an animated GIF.

    python make_gif.py                       # 10 s, racquets steered by the analytic fallback
    python make_gif.py --drex                # steered by the hosted Drex model (needs
                                             # NACE_API_KEY; runs in real time)
    python make_gif.py --seconds 6 --fps 20 --width 480 --out rally.gif

Needs Pillow for the GIF (pip install pillow). Without it, the frames are saved to
<out>.npz instead.
"""
import argparse
import os
import threading
import time
import types

import numpy as np
import mujoco

import pingpong_drex as P


def save_gif(frames, path, fps):
    try:
        from PIL import Image
    except ImportError:
        npz = os.path.splitext(path)[0] + ".npz"
        np.savez_compressed(npz, frames=np.stack(frames), fps=fps)
        print(f"Pillow isn't installed (pip install pillow), so the frames went to {npz}")
        return
    # One shared palette (taken from a frame with the ball in it) keeps colours stable.
    palette = Image.fromarray(frames[len(frames) // 2]).quantize(colors=128)
    imgs = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=round(1000 / fps),
                 loop=0, optimize=True, disposal=1)
    print(f"wrote {path}: {len(frames)} frames, {os.path.getsize(path) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description="Render the ping pong game to an animated GIF")
    ap.add_argument("--out", default="pingpong.gif")
    ap.add_argument("--seconds", type=float, default=10.0, help="length of the clip")
    ap.add_argument("--fps", type=int, default=30, choices=(15, 20, 30, 60))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--seed", type=int, default=7, help="serve randomness (try others for a different clip)")
    ap.add_argument("--drex", action="store_true", help="steer with the hosted Drex model (real time)")
    ap.add_argument("--model", default=P.HOSTED_MODEL)
    a = ap.parse_args()

    args = types.SimpleNamespace(
        no_drex=not a.drex, max_age=0.6, quiet=False, serve_speed=2.0, sim_speed=1.0,
        url=P.HOSTED_URL, api_key=os.environ.get("NACE_API_KEY"), model=a.model,
        timeout=10.0, min_interval=0.25)
    if a.drex and not args.api_key:
        ap.error("--drex needs the NACE_API_KEY environment variable")

    g = P.Game(args)
    g.rng = np.random.default_rng(a.seed)
    g.reset_and_serve()
    if a.drex:
        threading.Thread(target=g.worker, daemon=True).start()

    m, d = g.model, g.data
    renderer = mujoco.Renderer(m, height=a.height, width=a.width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.82]
    cam.azimuth, cam.elevation, cam.distance = 90, -18, 3.6

    game_hz = 60                                   # same control rate as the game
    steps = max(1, int(round(1 / game_hz / m.opt.timestep)))
    every = game_hz // a.fps                       # game frames per GIF frame
    frames = []
    n_game_frames = int(a.seconds * game_hz)
    t_start = time.time()
    for i in range(n_game_frames):
        g.advance(steps)
        if i % every == 0:
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())
        if a.drex:                                 # keep Drex's answers in step with play
            time.sleep(max(0.0, t_start + (i + 1) / game_hz - time.time()))
    g.stop.set()
    renderer.close()
    save_gif(frames, a.out, a.fps)


if __name__ == "__main__":
    main()
