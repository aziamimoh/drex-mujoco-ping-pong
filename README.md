# Drex MuJoCo Ping Pong

![A rally in the simulation](pingpong.gif)

Table tennis in [MuJoCo](https://mujoco.org/), with both racquets steered by a
[Drex](https://www.nace.ai/) decision model: the hosted Nace API by default, or a local
Drex server.

While the ball flies toward a racquet, a background thread asks Drex three questions:

- **lane:** where should the racquet move sideways?
- **style:** what kind of return? *lift* (high, safe arc), *flat* (low, fast, deep) or
  *block* (short and gentle).
- **stroke:** forehand or backhand?

The physics loop never waits for Drex. Every frame the script predicts where, when, how
fast and with what spin the ball will reach the racquet, accounting for gravity, table
bounces and spin. It then picks a return path that clears the net by the chosen style's
margin and lands at its depth. Finally it solves for the swing speed, blade tilt and
sideways blade turn that send the ball along that path. The table and blade bounce
constants are measured from the MuJoCo model at startup. If no fresh Drex answer is
available, the racquet falls back to the prediction and a lift.

## Requirements

- Python 3.10+
- `pip install mujoco numpy` (and `pillow` for `make_gif.py`)
- For the hosted Drex API, a Nace API key

## Run

From this folder (the scripts load `pingpong.xml` from the current directory):

```powershell
# Hosted Drex API (default model drex-max-v1.1)
$env:NACE_API_KEY = "nace_sk_..."
python pingpong_drex.py

# A local Drex server instead (no key needed)
python pingpong_drex.py --url http://127.0.0.1:8000/v1/systemone

# No model at all: the racquets use the prediction alone
python pingpong_drex.py --no-drex

# Slow motion (also gives Drex more time per decision)
python pingpong_drex.py --sim-speed 0.5
```

After every return the console prints what was aimed for and where the ball went, e.g.

```
R lift  forehand swing 1.30 m/s  tilt +0.59  yaw +0.00  (solver miss 0.00 m/s)  aimed x=-0.75 y=+0.00  -> landed at x=-0.79 y=-0.00
```

Run `python pingpong_drex.py --help` for all options: request timeout, how long an answer
stays valid, the minimum gap between Drex calls (the hosted API bills per request), serve
speed, and `--quiet`.

## Files

| File | What it is |
|---|---|
| `pingpong.xml` | The scene: a regulation-size table, net, 40 mm / 2.7 g ball, and two racquets that swing (x), slide (y, z), turn (yaw), tilt and roll |
| `pingpong_drex.py` | The game: Drex client, ball-flight prediction, stroke solver and racquet control |
| `pingpong_controller.py` | The original simple controller (straight-line aiming, no model) |
| `make_gif.py` | Renders a clip of the game straight from MuJoCo and saves it as an animated GIF (`python make_gif.py`, or `--drex` to steer with the hosted model) |
| `stroke_test.py` | Headless test: fires balls at each racquet and compares the planned stroke with what MuJoCo does |
| `bounce_test.py` | Headless test: measures ball bounce restitution for several contact settings |
| `debug.py`, `test_simulation.py` | Small checks that the model loads and the ball bounces |

## Notes

- Hinge-joint ranges in `pingpong.xml` are in **degrees** (MuJoCo's default unit for
  angles in XML), while joint positions and control values are in radians.
- The contact setting `solref="0.03 0.05"` gives bounces that keep about 86% of their
  speed. Much stiffer settings (such as `0.004 0.08`) make each contact too short for the
  0.5 ms timestep, and the ball gains energy on every bounce. Use `bounce_test.py` to
  check any change.
