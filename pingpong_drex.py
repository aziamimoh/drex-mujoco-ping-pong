"""
Ping pong in MuJoCo, with the racquets steered by a Drex model (hosted Nace API by
default, or a local server).

How it works
------------
* The physics loop runs at ~60 fps in the main thread and never waits on the model.
* While a ball is heading toward a racquet, a background thread sends Drex the ball
  state plus three typed questions (POST /v1/systemone):
    - lane:   which of 7 sideways lanes should the racquet move to?
    - style:  lift / flat / block return (how high over the net, how deep)?
    - stroke: forehand / backhand?
  The returned probabilities are turned into continuous targets (probability-weighted
  average of the lane positions / style settings).
* Every frame the code predicts where, when and how fast (and with what spin) the ball
  reaches the racquet, using gravity, table bounces and spin. It then picks a return
  path that clears the net by the style's margin and lands at the style's depth, and
  solves for the swing speed, blade tilt and blade turn (yaw) that send the ball along
  that path. Table and blade bounce constants are measured from the MuJoCo model at
  startup, so the solver uses the simulation's own physics.
* Each racquet pulls back and swings forward to meet the ball at the predicted moment.
  A backhand turns the blade so the handle leans the other way.
* If no fresh answer is available (model slow, server down, --no-drex), the racquet
  uses the analytic prediction and a default lift, so the game never stalls.
* After every hit the console prints what was aimed for and where the ball went.

Usage
-----
Hosted API (default). Put your key in an environment variable first (PowerShell):
    $env:NACE_API_KEY = "nace_sk_..."
    python pingpong_drex.py                                   # model drex-max-v1.1
    python pingpong_drex.py --model drex-v2.0                 # another hosted model

Local server instead (no key needed):
    python pingpong_drex.py --url http://127.0.0.1:8000/v1/systemone

    python pingpong_drex.py --no-drex             # analytic fallback only (no model needed)
    python pingpong_drex.py --sim-speed 0.5       # slow motion (also gives Drex more time)

The hosted API bills per request: --min-interval sets the gap between calls
(default 0.25 s, i.e. at most ~4 calls per second while a ball is in flight).
If answers are slow, give them more time:  --timeout 20 --max-age 1.5
"""
import argparse
import http.client
import json
import os
import threading
import time
import urllib.parse

import numpy as np
import mujoco
import mujoco.viewer

HOME_Z = 0.95            # racquet rest height (matches pingpong.xml)
RACQUET_X = 1.5          # racquet home distance from table center (|x|)
RESERVE_AFTER = 15.0     # seconds before forcing a re-serve
MAX_BALL_SPEED = 8.0     # m/s safety cap
GRAVITY = 9.81
TABLE_Z = 0.76           # table surface height
TABLE_HALF = (1.37, 0.7625)
NET_TOP = TABLE_Z + 0.1525
BALL_R = 0.02
BLADE_HALF_T = 0.008
BLADE_OFFSET = BLADE_HALF_T + BALL_R      # blade centre to ball centre at contact

# ---- swing ---------------------------------------------------------------------
# The racquet pulls back, then drives forward through the ball and meets it with its blade
# centre at |x| = CONTACT_RACQUET_X. The follow-through stops at X_MIN.
CONTACT_RACQUET_X = 1.45
X_MIN, X_MAX = 1.39, 1.66
SWING_TIME = 0.15        # max seconds of forward drive before contact (shorter for fast swings,
                         # so the backswing stays within X_MAX)
X_GAIN = 40.0            # position correction of the x velocity servo (1/s)
X_VMAX = 3.0             # cap on the commanded x velocity (m/s)
CONTACT_X = CONTACT_RACQUET_X - BLADE_OFFSET   # ball-centre |x| at contact
ROLL = 0.7               # rad the blade spins about its face for forehand/backhand
SWING_MAX = 2.8          # fastest swing the solver may ask for (m/s)
TILT_MAX = 0.95          # stay inside the joint ranges in pingpong.xml (tilt +-57 deg, yaw +-35 deg)
YAW_MAX = 0.5

# Bounce constants: fraction of the normal speed kept (restitution) and of the sliding speed
# kept by a ball with no spin (grip). Measured from the MuJoCo model at startup by
# calibrate(); these are only fallbacks (values bounce_test.py found for solref 0.03/0.05).
TABLE_RESTITUTION, TABLE_GRIP = 0.87, 0.92
BLADE_RESTITUTION, BLADE_GRIP = 0.85, 0.92
MAX_FLIGHT_TIME = 2.0    # longer predicted flights are treated as unplayable
HOSTED_URL = "https://console.nace.ai/v1/systemone"
HOSTED_MODEL = "drex-max-v1.1"

# ---- what Drex chooses between -------------------------------------------------
LANE_Y = {f"lane_{i + 1}": y for i, y in enumerate([-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9])}
# style -> (landing depth past the net in m, height the ball should clear the net by in m)
STYLE = {"lift": (0.75, 0.30), "flat": (0.95, 0.15), "block": (0.50, 0.15)}
FALLBACK_STYLE = "lift"  # used when no Drex answer is available
AIM_Y = 0.0              # sideways landing target on the opponent's half (0 = centre line)

QUESTIONS = {
    "lane": {
        "type": "choice",
        "instructions": (
            "A table tennis ball is flying toward the racquet. Choose the lane the racquet "
            "should move to so it meets the ball at the racquet plane. Match the lane to "
            "predicted_y_at_racquet_m."
        ),
        "criteria": {k: f"Racquet centered at y = {y:+.1f} m" for k, y in LANE_Y.items()},
    },
    "style": {
        "type": "choice",
        "instructions": (
            "Choose how to return the ball. Use lift when the ball will arrive low "
            "(see predicted_height_above_table_at_racquet_m) or slow, flat when it arrives "
            "well above the net, block when it arrives fast."
        ),
        "criteria": {
            "lift": "High, safe arc well over the net, landing mid-table",
            "flat": "Low, fast arc just over the net, landing deep",
            "block": "Short, gentle return landing close to the net",
        },
    },
    "stroke": {
        "type": "choice",
        "instructions": (
            "The player is right-handed. Use a forehand when the ball arrives on the player's "
            "right (see ball_arrives_on_player_side), and a backhand when it arrives on the "
            "player's left or straight at the body."
        ),
        "criteria": {
            "forehand": "Forehand stroke on the player's right side",
            "backhand": "Backhand stroke on the player's left side",
        },
    },
}


# ---- ball flight prediction ----------------------------------------------------
def predict_arrival(pos, vel, x_target, spin=(0.0, 0.0, 0.0)):
    """Ballistic flight with table bounces (no air drag; the net is ignored).
    spin is the ball's world-frame angular velocity: at each bounce, friction removes part of
    the slip between the ball's surface and the table, which changes both the horizontal speed
    and the spin. Returns (seconds, position, velocity, spin) of the ball at x_target, or None."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
    wx, wy, wz = float(spin[0]), float(spin[1]), float(spin[2])
    zc = TABLE_Z + BALL_R
    t = 0.0
    for _ in range(6):                      # at most a few bounces before arrival
        if abs(vx) < 1e-3:
            return None
        t_rem = (x_target - x) / vx
        if t_rem <= 0:
            return None
        if z < zc - 1e-3:                   # already below the table top: no bounce possible
            break
        s = (vz + np.sqrt(max(vz * vz + 2 * GRAVITY * (z - zc), 0.0))) / GRAVITY
        if s <= 1e-6 or s >= t_rem:         # reaches the racquet before landing
            break
        bx, by = x + vx * s, y + vy * s
        if abs(bx) > TABLE_HALF[0] or abs(by) > TABLE_HALF[1]:
            break                           # lands off the table: keeps falling
        x, y, z, t = bx, by, zc, t + s
        vz = -(vz - GRAVITY * s) * TABLE_RESTITUTION
        # contact-point slip velocity (bottom of the ball), and the friction impulse's effect
        ux, uy = vx - BALL_R * wy, vy + BALL_R * wx
        dvx, dvy = -(1 - TABLE_GRIP) * ux, -(1 - TABLE_GRIP) * uy
        vx, vy = vx + dvx, vy + dvy
        wx, wy = wx + 2.5 * dvy / BALL_R, wy - 2.5 * dvx / BALL_R   # solid sphere, I = 2/5 m r^2
    else:
        return None                         # still bouncing: a slow, nearly rolling ball
    if abs(vx) < 1e-3:
        return None
    t_rem = (x_target - x) / vx
    if t_rem <= 0 or t + t_rem > MAX_FLIGHT_TIME:
        return None
    p = np.array([x_target, y + vy * t_rem, z + vz * t_rem - 0.5 * GRAVITY * t_rem * t_rem])
    v = np.array([vx, vy, vz - GRAVITY * t_rem])
    return t + t_rem, p, v, np.array([wx, wy, wz])


# ---- the stroke: blade bounce model and solver ---------------------------------
def face_normal(side, tilt, yaw):
    """Unit normal of the racquet face that points toward the table, for given joint angles.
    The joints apply yaw (about z), then tilt (about y); roll spins about this normal."""
    ct, st, cy, sy = np.cos(tilt), np.sin(tilt), np.cos(yaw), np.sin(yaw)
    n0 = -side                                  # face normal before rotation: (-side, 0, 0)
    nx, nz = ct * n0, -st * n0                  # tilt about y
    return np.array([cy * nx, sy * nx, nz])     # then yaw about z


def blade_hit(vin, spin, n, racquet_vel):
    """Ball velocity after bouncing off a blade with face normal n moving at racquet_vel:
    the normal part of the relative velocity is reversed and scaled by BLADE_RESTITUTION,
    and friction removes part of the slip at the contact point (spin included)."""
    rel = vin - racquet_vel
    vn = rel @ n
    slip = rel - vn * n + np.cross(spin, -BALL_R * n)
    return vin - (1 + BLADE_RESTITUTION) * vn * n - (1 - BLADE_GRIP) * slip


def aim_velocity(p0, side, depth, clearance, y_land):
    """Velocity the ball must leave the racquet with to pass over the net with `clearance`
    to spare and land `depth` metres past the net on the other side. None if impossible."""
    d1 = side * p0[0]                       # horizontal distance to the net
    d2 = d1 + depth                         # ... to the landing point
    h1 = NET_TOP + BALL_R + clearance - p0[2]
    h2 = TABLE_Z + BALL_R - p0[2]
    # z(d) = z0 + a*d - b*d^2  with a = vz/vh and b = g / (2 vh^2): two equations, a and b
    det = -d1 * d2 * d2 + d2 * d1 * d1
    if abs(det) < 1e-9:
        return None
    a = (-h1 * d2 * d2 + h2 * d1 * d1) / det
    b = (d1 * h2 - d2 * h1) / det
    if b <= 1e-6:
        return None
    vh = np.sqrt(GRAVITY / (2 * b))
    t_land = d2 / vh
    return np.array([-side * vh, (y_land - p0[1]) / t_land, a * vh])


def solve_stroke(side, vin, spin, v_target, guess):
    """Find (swing speed, tilt, yaw) whose blade bounce sends the ball off at v_target.
    Damped Gauss-Newton with a numeric Jacobian, kept inside the joint limits.
    Returns (params, residual in m/s)."""
    lo = np.array([0.0, -TILT_MAX, -YAW_MAX])
    hi = np.array([SWING_MAX, TILT_MAX, YAW_MAX])

    def err(x):
        n = face_normal(side, x[1], x[2])
        return blade_hit(vin, spin, n, np.array([-side * x[0], 0.0, 0.0])) - v_target

    x = np.clip(np.asarray(guess, float), lo, hi)
    f = err(x)
    for _ in range(25):
        if np.linalg.norm(f) < 1e-3:
            break
        J = np.empty((3, 3))
        for k in range(3):
            dx = np.zeros(3)
            dx[k] = 1e-5
            J[:, k] = (err(x + dx) - f) / 1e-5
        step = np.linalg.lstsq(J, -f, rcond=None)[0]
        step *= min(1.0, 0.8 / max(abs(step[0]), 1e-9), 0.3 / max(np.abs(step[1:]).max(), 1e-9))
        for _ in range(6):                  # backtrack until the error shrinks
            xn = np.clip(x + step, lo, hi)
            fn = err(xn)
            if np.linalg.norm(fn) < np.linalg.norm(f):
                x, f = xn, fn
                break
            step *= 0.5
        else:
            break
    return x, float(np.linalg.norm(f))


# ---- calibration from the MuJoCo model ----------------------------------------
def _measure(model, pos, vel, coord, plane, sign):
    """Fire the ball in a scratch copy of the simulation and return its velocity just before
    and just after it crosses `plane` on axis `coord` (going in direction `sign`, then back)."""
    d = mujoco.MjData(model)
    j = model.joint("ball_free")
    q, v = j.qposadr[0], j.dofadr[0]
    d.qpos[q:q + 3] = pos
    d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
    d.qvel[v:v + 3] = vel
    mujoco.mj_forward(model, d)
    vin = None
    for _ in range(int(0.5 / model.opt.timestep)):
        before_c, before_v = d.qpos[q + coord], d.qvel[v:v + 3].copy()
        mujoco.mj_step(model, d)
        c = d.qpos[q + coord]
        if vin is None and sign * (before_c - plane) < 0 <= sign * (c - plane):
            vin = before_v
        elif vin is not None and sign * (before_c - plane) > 0 >= sign * (c - plane):
            return vin, d.qvel[v:v + 3].copy()
    return None


def calibrate(model):
    """Measure table and blade restitution/grip from the model's own contact settings."""
    global TABLE_RESTITUTION, TABLE_GRIP, BLADE_RESTITUTION, BLADE_GRIP
    zc = TABLE_Z + BALL_R
    r = _measure(model, [0.3, 0.0, zc + 0.05], [2.5, 0.0, -3.5], 2, zc + 0.002, -1)
    if r:
        vin, vout = r
        e, grip = vout[2] / -vin[2], vout[0] / vin[0]
        if 0.3 < e < 0.99 and 0.4 < grip <= 1.0:
            TABLE_RESTITUTION, TABLE_GRIP = float(e), float(grip)
        else:
            print(f"WARNING: table bounce restitution={e:.3f} grip={grip:.3f} "
                  "(a value >= 1 means bounces add energy; check solref in pingpong.xml)")
    face = RACQUET_X - BLADE_OFFSET                 # right racquet at rest
    r = _measure(model, [1.2, 0.0, 0.85], [3.0, 0.0, 1.5], 0, face - 0.002, +1)
    if r:
        vin, vout = r
        e, grip = -vout[0] / vin[0], vout[2] / vin[2]
        if 0.3 < e < 0.99 and 0.4 < grip <= 1.0:
            BLADE_RESTITUTION, BLADE_GRIP = float(e), float(grip)
        else:
            print(f"WARNING: blade bounce restitution={e:.3f} grip={grip:.3f}; using defaults")
    print(f"bounce constants: table e={TABLE_RESTITUTION:.3f} grip={TABLE_GRIP:.3f}  "
          f"blade e={BLADE_RESTITUTION:.3f} grip={BLADE_GRIP:.3f}")


# ---- Drex client ---------------------------------------------------------------
def build_state(snap):
    return {
        "game": "table tennis, ball flying toward one racquet",
        "racquet_side": "right" if snap["side"] > 0 else "left",
        "ball_position_m": [round(float(v), 3) for v in snap["pos"]],
        "ball_velocity_mps": [round(float(v), 3) for v in snap["vel"]],
        "ball_height_above_table_m": round(float(snap["pos"][2] - TABLE_Z), 3),
        "predicted_y_at_racquet_m": round(float(snap["pred_y"]), 3),
        "predicted_height_above_table_at_racquet_m": round(float(snap["pred_z"] - TABLE_Z), 3),
        "predicted_speed_at_racquet_mps": round(float(snap["pred_speed"]), 3),
        "player": "right-handed",
        "ball_arrives_on_player_side": ("right (forehand side)" if snap["pred_y"] * snap["side"] > 0
                                        else "left (backhand side)"),
        "seconds_until_ball_arrives": round(float(snap["eta"]), 3),
        "net_height_above_table_m": 0.1525,
    }


def _expected(probs, values):
    total = sum(probs.get(k, 0.0) for k in values)
    if total <= 0:
        return None
    return sum(probs.get(k, 0.0) * v for k, v in values.items()) / total


class DrexClient:
    """POSTs to a System One endpoint over one kept-alive connection (saves a TLS
    handshake per call on the hosted API). Used only from the worker thread."""

    def __init__(self, url, api_key, model, timeout):
        u = urllib.parse.urlsplit(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError(f"bad --url: {url}")
        self.https = u.scheme == "https"
        self.host = u.hostname
        self.port = u.port
        self.path = u.path or "/"
        self.model = model
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json", "X-Nace-Client": "pingpong-mujoco/1"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.conn = None

    def _connect(self):
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        self.conn = cls(self.host, self.port, timeout=self.timeout)

    def _close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def post(self, payload):
        if self.model:
            payload = {**payload, "model": self.model}
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(2):
            reused = self.conn is not None
            if not reused:
                self._connect()
            try:
                self.conn.request("POST", self.path, body=body, headers=self.headers)
                resp = self.conn.getresponse()
                data = resp.read()
            except (http.client.HTTPException, ConnectionError):
                self._close()
                if reused and attempt == 0:      # stale keep-alive connection: retry once
                    continue
                raise
            except Exception:
                self._close()
                raise
            if resp.status != 200:
                if resp.status in (401, 403):
                    raise RuntimeError(f"HTTP {resp.status} (check NACE_API_KEY): {data[:200]!r}")
                raise RuntimeError(f"HTTP {resp.status}: {data[:300]!r}")
            return json.loads(data)


def ask_drex(client, snap):
    """One System One request.
    Returns dict(y, depth, clearance, backhand, lane, style, stroke)."""
    answers = client.post({"state": build_state(snap), "questions": QUESTIONS})["answers"]

    lane_p = answers["lane"]["probabilities"]
    style_p = answers["style"]["probabilities"]
    stroke_p = answers["stroke"]["probabilities"]
    y = _expected(lane_p, LANE_Y)
    depth = _expected(style_p, {k: v[0] for k, v in STYLE.items()})
    clearance = _expected(style_p, {k: v[1] for k, v in STYLE.items()})
    p_back = _expected(stroke_p, {"forehand": 0.0, "backhand": 1.0})
    if y is None or depth is None or p_back is None:
        raise ValueError("unexpected answer shape: %r" % (answers,))
    return dict(y=y, depth=depth, clearance=clearance, backhand=p_back > 0.5,
                lane=answers["lane"].get("choice"), style=answers["style"].get("choice"),
                stroke=answers["stroke"].get("choice"))


# ---- game ----------------------------------------------------------------------
class Game:
    def __init__(self, args):
        self.args = args
        self.model = mujoco.MjModel.from_xml_path("pingpong.xml")
        self.data = mujoco.MjData(self.model)
        self.bq = self.model.joint("ball_free").qposadr[0]
        self.bv = self.model.joint("ball_free").dofadr[0]
        self.ball_body = self.model.body("ball").id
        act = lambda n: self.model.actuator(n).id
        self.paddles = {}
        for side, name in ((-1, "left"), (+1, "right")):
            self.paddles[side] = dict(
                x=act(f"{name}_x_ctrl"), y=act(f"{name}_y_ctrl"), z=act(f"{name}_z_ctrl"),
                yaw=act(f"{name}_yaw_ctrl"), tilt=act(f"{name}_tilt_ctrl"),
                roll=act(f"{name}_roll_ctrl"), xq=self.model.joint(f"{name}_x").qposadr[0])
        calibrate(self.model)
        self.plans = {-1: None, +1: None}   # side -> swing plan dict(t_c, v) or None
        self.aims = {-1: None, +1: None}    # side -> latest stroke plan, for the shot log
        self.guess = {-1: None, +1: None}   # side -> (shot, solver solution) warm start
        self.rng = np.random.default_rng()
        self.lock = threading.Lock()
        self.snapshot = None          # latest ball snapshot (written by main thread)
        self.decisions = {}           # side -> decision dict (written by worker)
        self.stop = threading.Event()
        self.stats = dict(calls=0, errors=0, latency=0.0, used=0, fallback=0, last_error="")
        # Shot counter: +1 on every serve and every time the ball changes direction in x.
        # Drex answers carry the shot they were asked about, so a late answer is never
        # applied to a different ball.
        self.shot = 0
        self.last_dir = 0
        self.watch = None             # the return being followed for the shot log
        self.rolling_since = None     # sim time the ball started rolling on the table
        self.too_fast_reported = False

    # -- serving / safety --
    def serve(self):
        d = self.data
        side = self.rng.choice([-1, 1])
        d.qpos[self.bq:self.bq + 3] = [-side * 1.0, self.rng.uniform(-0.3, 0.3), 1.15]
        d.qpos[self.bq + 3:self.bq + 7] = [1, 0, 0, 0]
        d.qvel[self.bv:self.bv + 3] = [side * self.args.serve_speed, self.rng.uniform(-0.3, 0.3), 0.0]
        d.qvel[self.bv + 3:self.bv + 6] = 0
        self.shot += 1
        self.last_dir = int(side)
        self.decisions.clear()
        self.plans = {-1: None, +1: None}
        self.watch = None
        self.too_fast_reported = False
        self.rolling_since = None
        mujoco.mj_forward(self.model, d)

    def reset_and_serve(self):
        mujoco.mj_resetData(self.model, self.data)
        self.serve()

    # -- ball tracking --
    @staticmethod
    def heading_to(side, pos, vel):
        """True while the ball flies toward this side's racquet and is still far enough
        away for a Drex answer to be useful (used only to decide when to ask Drex)."""
        return (vel[0] * side) > 0.05 and (pos[0] * side) < RACQUET_X - 0.1

    @staticmethod
    def engaged(side, pos, vel):
        """True until the ball is returned (x velocity flips) or has passed the racquet.
        The racquet keeps tracking right up to contact."""
        return (vel[0] * side) > 0.05 and (pos[0] * side) < RACQUET_X + 0.05

    def ball_spin(self):
        """Ball angular velocity in the world frame (the free joint stores it in the ball's
        own frame)."""
        w_local = self.data.qvel[self.bv + 3:self.bv + 6]
        return self.data.xmat[self.ball_body].reshape(3, 3) @ w_local

    def track_direction(self):
        d = self.data
        vx = d.qvel[self.bv]
        direction = 1 if vx > 0.05 else -1 if vx < -0.05 else 0
        if direction and direction != self.last_dir:
            self.shot += 1
            self.last_dir = direction
            hitter = -direction
            if abs(d.qpos[self.bq]) > 1.2 and d.qpos[self.bq] * hitter > 0:
                self.start_watch(hitter)

    def make_snapshot(self):
        pos = self.data.qpos[self.bq:self.bq + 3].copy()
        vel = self.data.qvel[self.bv:self.bv + 3].copy()
        side = 0
        if self.heading_to(+1, pos, vel):
            side = +1
        elif self.heading_to(-1, pos, vel):
            side = -1
        if side == 0:
            return None
        pred = predict_arrival(pos, vel, side * CONTACT_X, self.ball_spin())
        if pred is None:
            return None
        eta, p, v, _ = pred
        return dict(side=side, shot=self.shot, pos=pos, vel=vel, eta=eta,
                    pred_y=p[1], pred_z=p[2], pred_speed=np.linalg.norm(v))

    # -- background worker: asks Drex --
    def worker(self):
        client = DrexClient(self.args.url, self.args.api_key, self.args.model, self.args.timeout)
        while not self.stop.is_set():
            t0 = time.time()
            with self.lock:
                snap = self.snapshot
            if snap is None:
                time.sleep(0.03)
                continue
            try:
                dec = ask_drex(client, snap)
                dec["t"] = time.time()
                dec["shot"] = snap["shot"]
                self.decisions[snap["side"]] = dec
                self.stats["calls"] += 1
                if self.stats["calls"] == 1:
                    print(f"first Drex answer received in {time.time() - t0:.2f} s: "
                          f"lane={dec['lane']} style={dec['style']} stroke={dec['stroke']}")
                self.stats["latency"] = 0.8 * self.stats["latency"] + 0.2 * (time.time() - t0)
            except Exception as exc:     # server down, bad answer, timeout...
                self.stats["errors"] += 1
                self.stats["last_error"] = f"{type(exc).__name__}: {exc}"
                if self.stats["errors"] == 1:
                    print(f"first Drex error: {self.stats['last_error']}")
                time.sleep(0.5)
            time.sleep(max(0.0, self.args.min_interval - (time.time() - t0)))

    # -- racquet control (main thread) --
    def control(self, side):
        d = self.data
        p = self.paddles[side]
        pos = d.qpos[self.bq:self.bq + 3]
        vel = d.qvel[self.bv:self.bv + 3]
        if not self.engaged(side, pos, vel):
            for k in ("y", "z", "yaw", "tilt", "roll"):
                d.ctrl[p[k]] = 0.0
            self.plans[side] = None
            return

        pred = predict_arrival(pos, vel, side * CONTACT_X, self.ball_spin())
        if pred is None:                    # unplayable for now: follow the ball, no swing
            d.ctrl[p["y"]] = np.clip(pos[1], -0.9, 0.9)
            d.ctrl[p["z"]] = np.clip(pos[2] - HOME_Z, -0.2, 0.4)
            self.plans[side] = None
            return
        t_arr, pb, vb, wb = pred

        # The player's right is -y for the left racquet and +y for the right one, so
        # y * side > 0 is the forehand side; balls near the middle are taken on the backhand.
        style = FALLBACK_STYLE
        depth, clearance = STYLE[style]
        target_y = pb[1]
        backhand = pb[1] * side < 0.1
        dec = self.decisions.get(side)
        if (dec and not self.args.no_drex and dec.get("shot") == self.shot
                and time.time() - dec["t"] < self.args.max_age):
            target_y, depth, clearance = dec["y"], dec["depth"], dec["clearance"]
            backhand, style = dec["backhand"], dec["style"]
            self.stats["used"] += 1
        else:
            self.stats["fallback"] += 1

        # Return path -> swing speed, tilt and yaw that produce it.
        v_target = aim_velocity(pb, side, depth, clearance, AIM_Y)
        if v_target is None:                # e.g. contact far above the net: just push it back
            v_target = np.array([-side * 3.0, -0.3 * pb[1], 0.5])
        g = self.guess[side]
        guess = g[1] if g and g[0] == self.shot else (1.5, side * 0.4, 0.0)
        (u, tilt, yaw), resid = solve_stroke(side, vb, wb, v_target, guess)
        self.guess[side] = (self.shot, (u, tilt, yaw))
        n = face_normal(side, tilt, yaw)

        # Blade centre sits BLADE_OFFSET behind the ball centre along the face normal.
        d.ctrl[p["y"]] = np.clip(target_y - BLADE_OFFSET * n[1], -0.9, 0.9)
        d.ctrl[p["z"]] = np.clip(pb[2] - BLADE_OFFSET * n[2] - HOME_Z, -0.2, 0.4)
        d.ctrl[p["yaw"]] = yaw
        d.ctrl[p["tilt"]] = tilt
        # Roll spins the blade about its face normal. A positive roll leans the handle toward +y.
        # Forehand: handle leans toward the player's body, i.e. their left (sign -side);
        # backhand: toward their right (sign +side).
        d.ctrl[p["roll"]] = (side if backhand else -side) * ROLL
        self.plans[side] = dict(t_c=d.time + t_arr, v=max(u, 0.05))
        self.aims[side] = dict(style=style, stroke="backhand" if backhand else "forehand",
                               u=u, tilt=tilt, yaw=yaw, resid=resid, depth=depth,
                               v_target=v_target, normal=n, t_c=d.time + t_arr,
                               land_x=-side * depth, land_y=AIM_Y)

    def swing_target(self, side, t):
        """Desired racquet |x| and d|x|/dt at sim time t: wait pulled back, then drive forward
        at the planned speed so the blade centre crosses CONTACT_RACQUET_X exactly when the
        ball arrives, and follow through until X_MIN."""
        plan = self.plans[side]
        if plan is None:
            return RACQUET_X, 0.0
        v, t_c = plan["v"], plan["t_c"]
        back = min(CONTACT_RACQUET_X + v * SWING_TIME, X_MAX)
        t_start = t_c - (back - CONTACT_RACQUET_X) / v
        if t < t_start:
            return back, 0.0
        absx = CONTACT_RACQUET_X - v * (t - t_c)
        if absx <= X_MIN:
            return X_MIN, 0.0
        return absx, -v

    def drive_x(self):
        """Velocity command for both swing servos; called before every physics step."""
        d = self.data
        for side in (-1, +1):
            p = self.paddles[side]
            absx, vabs = self.swing_target(side, d.time)
            q_des = side * (absx - RACQUET_X)          # joint coordinate of that |x|
            cmd = side * vabs + X_GAIN * (q_des - d.qpos[p["xq"]])
            d.ctrl[p["x"]] = np.clip(cmd, -X_VMAX, X_VMAX)

    # -- shot log: what was aimed for, and where the return went --
    def start_watch(self, hitter):
        if self.args.quiet:
            return
        if self.watch:                      # previous return reached the racquet: it was good
            self.report("reached the other racquet")
        aim = self.aims.get(hitter)
        self.watch = dict(side=hitter, aim=aim, crossed=False,
                          x=self.data.qpos[self.bq], z=self.data.qpos[self.bq + 2],
                          vz=self.data.qvel[self.bv + 2])

    def report(self, outcome):
        w, self.watch = self.watch, None
        a = w["aim"]
        name = "R" if w["side"] > 0 else "L"
        if a is None:
            print(f"{name} return: {outcome}")
            return
        print(f"{name} {a['style']:5s} {a['stroke']:8s} swing {a['u']:.2f} m/s  tilt {a['tilt']:+.2f}"
              f"  yaw {a['yaw']:+.2f}  (solver miss {a['resid']:.2f} m/s)  aimed x={a['land_x']:+.2f}"
              f" y={a['land_y']:+.2f}  -> {outcome}")

    def check_watch(self):
        """Called after every physics step while a return is in flight."""
        w = self.watch
        if w is None:
            return
        d = self.data
        x, y, z = d.qpos[self.bq:self.bq + 3]
        vz = d.qvel[self.bv + 2]
        s = w["side"]
        if not w["crossed"] and x * s <= 0:          # crossing the net plane
            w["crossed"] = True
            if z < NET_TOP + BALL_R:
                self.report(f"net (crossed {z - NET_TOP:+.2f} m vs. net top)")
                return
        if w["vz"] < 0 <= vz and z < TABLE_Z + BALL_R + 0.01:   # bounced on something low
            if abs(x) <= TABLE_HALF[0] and abs(y) <= TABLE_HALF[1]:
                where = "own side" if x * s > 0 else "landed"
                self.report(f"{where} at x={x:+.2f} y={y:+.2f}")
                return
        if x * s < -TABLE_HALF[0] and z < TABLE_Z:
            self.report(f"long/wide (passed the far end at y={y:+.2f})")
            return
        if abs(y) > TABLE_HALF[1] + 0.05 and z < TABLE_Z:
            self.report(f"wide (y={y:+.2f})")
            return
        w["x"], w["z"], w["vz"] = x, z, vz

    # -- main loop --
    def run(self):
        self.reset_and_serve()
        last_serve = time.time()
        last_report = time.time()
        steps = max(1, int(round(self.args.sim_speed / 60 / self.model.opt.timestep)))

        if not self.args.no_drex:
            threading.Thread(target=self.worker, daemon=True).start()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.lookat[:] = [0, 0, 0.8]
            viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 90, -20, 4.5

            while viewer.is_running():
                t0 = time.time()

                with self.lock:
                    self.snapshot = self.make_snapshot()
                self.control(-1)
                self.control(+1)

                for _ in range(steps):
                    self.drive_x()
                    mujoco.mj_step(self.model, self.data)
                    self.check_watch()
                self.track_direction()

                # safety checks on the ball
                vel = self.data.qvel[self.bv:self.bv + 3]
                speed = np.linalg.norm(vel)
                if not np.isfinite(speed) or not np.all(np.isfinite(self.data.qpos)):
                    print("simulation went NaN, resetting")
                    self.reset_and_serve()
                    last_serve = time.time()
                elif speed > MAX_BALL_SPEED:
                    if not self.too_fast_reported:
                        print(f"ball too fast: {speed:.1f} m/s (capped at {MAX_BALL_SPEED})")
                        self.too_fast_reported = True
                    vel *= MAX_BALL_SPEED / speed

                p = self.data.qpos[self.bq:self.bq + 3]
                # A ball rolling along the table (resting on it, no vertical motion) is dead.
                on_table = (p[2] < TABLE_Z + BALL_R + 0.005 and abs(vel[2]) < 0.1
                            and abs(p[0]) < TABLE_HALF[0] and abs(p[1]) < TABLE_HALF[1])
                if not on_table:
                    self.rolling_since = None
                elif self.rolling_since is None:
                    self.rolling_since = self.data.time
                rolling = on_table and self.data.time - self.rolling_since > 0.5
                if (p[2] < 0.1 or abs(p[0]) > 2.5 or abs(p[1]) > 1.5 or rolling
                        or time.time() - last_serve > RESERVE_AFTER):
                    if self.watch:
                        self.report("rolling on the table" if rolling else "out of play")
                    elif rolling and not self.args.quiet:
                        print("ball rolling on the table: new serve")
                    self.serve()
                    last_serve = time.time()

                if time.time() - last_report > 5.0:
                    s = self.stats
                    print(f"drex calls={s['calls']} errors={s['errors']} "
                          f"latency={s['latency'] * 1000:.0f} ms  "
                          f"frames with drex={s['used']} fallback={s['fallback']}"
                          + (f"  last error: {s['last_error']}" if s["errors"] else ""))
                    last_report = time.time()

                viewer.sync()
                time.sleep(max(0, 1 / 60 - (time.time() - t0)))

        self.stop.set()


def main():
    ap = argparse.ArgumentParser(description="MuJoCo ping pong steered by a Drex model")
    ap.add_argument("--url", default=HOSTED_URL,
                    help=f"System One endpoint (default: hosted {HOSTED_URL}; "
                         "local: http://127.0.0.1:8000/v1/systemone)")
    ap.add_argument("--api-key", default=os.environ.get("NACE_API_KEY"),
                    help="Nace API key (default: the NACE_API_KEY environment variable)")
    ap.add_argument("--model", default=None,
                    help=f"model name to send (default: {HOSTED_MODEL} for the hosted API, "
                         "none for a local server)")
    ap.add_argument("--timeout", type=float, default=10.0, help="per-request timeout in seconds")
    ap.add_argument("--max-age", type=float, default=0.6,
                    help="seconds a Drex answer stays valid")
    ap.add_argument("--min-interval", type=float, default=0.25,
                    help="minimum seconds between Drex calls (the hosted API bills per call)")
    ap.add_argument("--serve-speed", type=float, default=2.0,
                    help="ball serve speed in m/s (below ~1.5 the ball may not clear the net)")
    ap.add_argument("--sim-speed", type=float, default=1.0,
                    help="simulated seconds per real second (0.5 = half-speed slow motion)")
    ap.add_argument("--quiet", action="store_true", help="don't print a line per return")
    ap.add_argument("--no-drex", action="store_true", help="use the analytic fallback only")
    args = ap.parse_args()

    hosted = urllib.parse.urlsplit(args.url).hostname == urllib.parse.urlsplit(HOSTED_URL).hostname
    if hosted and args.model is None:
        args.model = HOSTED_MODEL
    if hosted and not args.api_key and not args.no_drex:
        ap.error("the hosted API needs a key: set NACE_API_KEY or pass --api-key "
                 "(or use --no-drex / a local --url)")
    if not args.no_drex:
        print(f"Drex: {args.url}  model={args.model or '(server default)'}")
    Game(args).run()


if __name__ == "__main__":
    main()
