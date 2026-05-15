#!/usr/bin/env python
"""
Capstone Design - CARLA 0.9.16 Final
======================================
자율주행: CARLA 기본 autopilot (차선유지 안정)
전방 감지: 같은 차선만 (횡방향 필터) → 감속 / 긴급정지
후방 감지: 같은 차선만 → 속도 높여 거리 확보
측면 감지: 실제 끼어들기만 (3조건) → 속도 조절로 양보
충돌 카운트: 차량/보행자만 (신호등 제외)
센서: 전/좌/우 YOLOv8 | LiDAR | 후방카메라 | Semantic LiDAR
HUD: 속도계 / 주행상태 / 360도레이더 / 날씨 / 충돌 / FPS
NPC: 차량 15대 + 보행자 10명

실행:
  python carla_project1_final.py
  python carla_project1_final.py --async

조작: ESC / Q 종료
"""

import os, sys, math, time, random, threading, collections, argparse

# agents 경로 자동 설정
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, 'PythonAPI', 'carla'))

import carla
import numpy as np

try:
    import pygame
    from pygame.locals import K_ESCAPE, K_q
except ImportError:
    raise RuntimeError('pip install pygame')

# ══════════════════════════════════════════════════════════════════
# YOLO 공유 모델
# ══════════════════════════════════════════════════════════════════
_yolo_model = None
_yolo_lock  = threading.Lock()

def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        print("[YOLO] YOLOv8n 모델 로딩 중...")
        _yolo_model = YOLO('yolov8n.pt')
        print("[YOLO] 모델 준비 완료.")
    return _yolo_model


# ══════════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════════
WEATHER_CYCLE = [
    ("Clear Noon",  carla.WeatherParameters.ClearNoon),
    ("Cloudy Noon", carla.WeatherParameters.CloudyNoon),
    ("Wet Noon",    carla.WeatherParameters.WetNoon),
    ("Soft Rain",   carla.WeatherParameters.SoftRainNoon),
    ("Hard Rain",   carla.WeatherParameters.HardRainNoon),
    ("Foggy",       carla.WeatherParameters(
                        fog_density=60, fog_distance=10,
                        sun_altitude_angle=45)),
]
WEATHER_INTERVAL = 12.0

# 전방 감지
FRONT_HALF_ANGLE = 20.0   # 전방 ±20도
STOP_DIST        = 8.0
SLOW_DIST        = 20.0

# 후방 감지
REAR_HALF_ANGLE  = 20.0   # 후방 ±20도 (180도 기준)
REAR_DIST        = 12.0

# 측면 감지 (끼어들기 전용)
SIDE_DIST        = 6.0    # 측면 감지 반경
SIDE_HALF_ANGLE  = 10.0   # 측면 ±10도 (90도 기준, 매우 좁게)
LANE_WIDTH       = 3.5    # 차선 폭 (m)

# 충돌 무시 접두사
IGNORE_COLLISION = (
    'traffic.traffic_light', 'traffic.stop',
    'traffic.yield', 'traffic.speed_limit', 'static.',
)


# ══════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════
class CustomTimer:
    def __init__(self): self.timer = time.perf_counter
    def time(self): return self.timer()

def get_speed_kmh(v):
    vel = v.get_velocity()
    return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def draw_text(surf, text, pos, font, color=(255,255,255), bg=None):
    lbl = font.render(text, True, color)
    if bg:
        pygame.draw.rect(surf, bg,
            (pos[0]-4, pos[1]-2, lbl.get_width()+8, lbl.get_height()+4))
    surf.blit(lbl, pos)

def angle_diff(a, b):
    """두 각도의 차이 -180~180"""
    return (a - b + 180) % 360 - 180

def get_fwd_lat(ego_yaw_rad):
    """전방/횡방향 단위벡터 반환"""
    fwd = (math.cos(ego_yaw_rad), math.sin(ego_yaw_rad))
    lat = (-math.sin(ego_yaw_rad), math.cos(ego_yaw_rad))
    return fwd, lat


# ══════════════════════════════════════════════════════════════════
# 감지 결과
# ══════════════════════════════════════════════════════════════════
class DetResult:
    def __init__(self):
        self.actor = None
        self.dist  = 999.0
    def clear(self):
        self.actor = None
        self.dist  = 999.0


# ══════════════════════════════════════════════════════════════════
# SafeAutopilot
# ══════════════════════════════════════════════════════════════════
class SafeAutopilot:
    ST_DRIVING = "자율주행 중"
    ST_SLOWING = "전방 감속 중"
    ST_STOPPED = "전방 긴급 정지!"
    ST_REAR    = "후방 차량 접근!"
    ST_MERGE   = "끼어들기 감지!"

    def __init__(self, vehicle, world, tm):
        self.vehicle      = vehicle
        self.world        = world
        self.tm           = tm
        self.status       = self.ST_DRIVING
        self._braking     = False

        # 감지 결과
        self.det_front = DetResult()
        self.det_rear  = DetResult()
        self.det_left  = DetResult()
        self.det_right = DetResult()

        # Traffic Manager 설정
        vehicle.set_autopilot(True, tm.get_port())
        tm.ignore_lights_percentage(vehicle, 0)
        tm.distance_to_leading_vehicle(vehicle, 3.0)
        tm.vehicle_percentage_speed_difference(vehicle, -10)
        tm.auto_lane_change(vehicle, False)
        tm.random_left_lanechange_percentage(vehicle, 0)
        tm.random_right_lanechange_percentage(vehicle, 0)
        tm.keep_slow_lane_rule_percentage(vehicle, 100)
        tm.set_global_distance_to_leading_vehicle(2.5)
        print("[+] Autopilot 활성화 (차선유지 강화 / 차선변경 완전 금지)")

    # ── 내부 헬퍼 ────────────────────────────────────────────────
    def _ego_vectors(self):
        tf  = self.vehicle.get_transform()
        yaw = math.radians(tf.rotation.yaw)
        fwd, lat = get_fwd_lat(yaw)
        return tf.location, tf.rotation.yaw, fwd, lat

    def _fwd_lat_dist(self, loc, aloc, fwd, lat):
        dx = aloc.x - loc.x
        dy = aloc.y - loc.y
        return dx*fwd[0]+dy*fwd[1], abs(dx*lat[0]+dy*lat[1])

    def _is_same_lane(self, aloc, loc, lat):
        """횡방향 거리가 차선 절반 이내 → 같은 차선"""
        dx = aloc.x - loc.x
        dy = aloc.y - loc.y
        return abs(dx*lat[0]+dy*lat[1]) < (LANE_WIDTH / 2.0)

    def _is_merging(self, actor, loc, yaw_deg, fwd, lat):
        """
        실제 끼어들기 판별 (3조건 중 2개 이상)
        1. 횡방향 거리 < LANE_WIDTH (차선 간격이 좁아짐)
        2. 전방 위치가 에고 근방 (-5m ~ +10m)
        3. 속도 벡터가 에고 쪽으로 향함
        """
        aloc = actor.get_location()
        fwd_d, lat_d = self._fwd_lat_dist(loc, aloc, fwd, lat)
        score = 0

        # 조건1: 차선 간격이 좁음
        if lat_d < LANE_WIDTH:
            score += 1

        # 조건2: 전방 근방에 위치 (너무 뒤에 있는 차는 제외)
        if -5.0 < fwd_d < 10.0:
            score += 1

        # 조건3: 속도 벡터가 에고 방향으로
        try:
            av = actor.get_velocity()
            # 액터 속도의 횡방향 성분
            av_lat = av.x*lat[0] + av.y*lat[1]
            # 에고 기준 액터의 방향 부호 (좌=-1, 우=+1)
            actor_side = 1.0 if (aloc.x-loc.x)*lat[0]+(aloc.y-loc.y)*lat[1] > 0 else -1.0
            # 액터가 에고 쪽으로 이동 중이면 score+1
            if av_lat * (-actor_side) > 0.5:
                score += 1
        except Exception:
            pass

        return score >= 2

    # ── 방향별 스캔 ──────────────────────────────────────────────
    def _scan(self):
        self.det_front.clear()
        self.det_rear.clear()
        self.det_left.clear()
        self.det_right.clear()

        loc, yaw_deg, fwd, lat = self._ego_vectors()

        # 탐색 대상: 보행자 + 차량
        targets  = list(self.world.get_actors().filter('walker.pedestrian.*'))
        targets += [a for a in self.world.get_actors().filter('vehicle.*')
                    if a.id != self.vehicle.id]

        for actor in targets:
            aloc  = actor.get_location()
            dx    = aloc.x - loc.x
            dy    = aloc.y - loc.y
            dist  = math.sqrt(dx*dx + dy*dy)
            if dist < 0.3:
                continue

            # 에고 기준 각도
            abs_angle  = math.degrees(math.atan2(dy, dx))
            rel_angle  = angle_diff(abs_angle, yaw_deg)
            is_ped     = 'walker' in actor.type_id

            # ── 전방 스캔 ─────────────────────────────────────────
            if abs(rel_angle) <= FRONT_HALF_ANGLE and dist <= SLOW_DIST:
                # 보행자는 항상 감지, 차량은 같은 차선만
                if is_ped or self._is_same_lane(aloc, loc, lat):
                    if dist < self.det_front.dist:
                        self.det_front.actor = actor
                        self.det_front.dist  = dist

            # ── 후방 스캔 ─────────────────────────────────────────
            elif abs(angle_diff(rel_angle, 180)) <= REAR_HALF_ANGLE \
                    and dist <= REAR_DIST:
                if is_ped or self._is_same_lane(aloc, loc, lat):
                    if dist < self.det_rear.dist:
                        self.det_rear.actor = actor
                        self.det_rear.dist  = dist

            # ── 측면 스캔 (끼어들기 전용) ─────────────────────────
            elif dist <= SIDE_DIST:
                # 좌측: rel_angle ~ -90도
                if abs(angle_diff(rel_angle, -90)) <= SIDE_HALF_ANGLE:
                    if self._is_merging(actor, loc, yaw_deg, fwd, lat):
                        if dist < self.det_left.dist:
                            self.det_left.actor = actor
                            self.det_left.dist  = dist
                # 우측: rel_angle ~ +90도
                elif abs(angle_diff(rel_angle, 90)) <= SIDE_HALF_ANGLE:
                    if self._is_merging(actor, loc, yaw_deg, fwd, lat):
                        if dist < self.det_right.dist:
                            self.det_right.actor = actor
                            self.det_right.dist  = dist

    # ── 메인 스텝 ─────────────────────────────────────────────────
    def run_step(self, dt=0.05):
        # 전방위 스캔
        self._scan()

        # ── 전방 긴급 정지 ────────────────────────────────────────
        if self.det_front.actor and self.det_front.dist <= STOP_DIST:
            self.status   = self.ST_STOPPED
            self._braking = True
            self.vehicle.set_autopilot(False)
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            return

        # ── 전방 감속 ─────────────────────────────────────────────
        if self.det_front.actor and self.det_front.dist <= SLOW_DIST:
            self.status   = self.ST_SLOWING
            self._braking = False
            self.vehicle.set_autopilot(True, self.tm.get_port())
            ratio    = (self.det_front.dist - STOP_DIST) / (SLOW_DIST - STOP_DIST)
            spd_diff = int((1.0 - ratio) * 65)
            self.tm.vehicle_percentage_speed_difference(self.vehicle, spd_diff)
            return

        # ── 측면 끼어들기 → 속도 줄여서 자연스럽게 양보 ────────────
        side = self.det_left if self.det_left.actor else self.det_right
        if side.actor:
            self.status = self.ST_MERGE
            self.vehicle.set_autopilot(True, self.tm.get_port())
            self.tm.vehicle_percentage_speed_difference(self.vehicle, 30)
            self._braking = False
            return

        # ── 후방 차량 접근 ────────────────────────────────────────
        if self.det_rear.actor and self.det_rear.dist <= REAR_DIST:
            self.status   = self.ST_REAR
            self._braking = False
            self.vehicle.set_autopilot(True, self.tm.get_port())
            self.tm.vehicle_percentage_speed_difference(self.vehicle, -30)
            return

        # ── 정상 주행 ─────────────────────────────────────────────
        self.status = self.ST_DRIVING
        if self._braking:
            self._braking = False
            self.vehicle.set_autopilot(True, self.tm.get_port())
            self.tm.auto_lane_change(self.vehicle, False)
            self.tm.random_left_lanechange_percentage(self.vehicle, 0)
            self.tm.random_right_lanechange_percentage(self.vehicle, 0)
            self.tm.keep_slow_lane_rule_percentage(self.vehicle, 100)
        self.tm.vehicle_percentage_speed_difference(self.vehicle, -10)

    def all_detections(self):
        """HUD/렌더링용 전체 감지 결과"""
        res = {}
        for name, det in [('FRONT', self.det_front), ('REAR', self.det_rear),
                          ('LEFT',  self.det_left),  ('RIGHT', self.det_right)]:
            if det.actor:
                res[name] = det
        return res


# ══════════════════════════════════════════════════════════════════
# DisplayManager
# ══════════════════════════════════════════════════════════════════
class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('Capstone CARLA Final Demo')
        self.grid_size   = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]),
                int(self.window_size[1]/self.grid_size[0])]

    def get_display_offset(self, gp):
        ds = self.get_display_size()
        return [int(gp[1]*ds[0]), int(gp[0]*ds[1])]

    def add_sensor(self, s): self.sensor_list.append(s)
    def render(self):
        if not self.display: return
        for s in self.sensor_list: s.render()
    def destroy(self):
        for s in self.sensor_list: s.destroy()
    def render_enabled(self): return self.display is not None


# ══════════════════════════════════════════════════════════════════
# SensorManager
# ══════════════════════════════════════════════════════════════════
class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform,
                 attached, sensor_options, display_pos, label=''):
        self.surface        = None
        self.world          = world
        self.display_man    = display_man
        self.display_pos    = display_pos
        self.sensor_options = sensor_options
        self.label          = label
        self.timer          = CustomTimer()
        self._font = pygame.font.SysFont('consolas', 13, bold=True)
        self.sensor = self._init_sensor(sensor_type, transform, attached, sensor_options)
        self.display_man.add_sensor(self)

    def _init_sensor(self, st, transform, attached, opts):
        ds = self.display_man.get_display_size()
        if st == 'RGBCamera':
            bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', str(ds[0]))
            bp.set_attribute('image_size_y', str(ds[1]))
            for k,v in opts.items(): bp.set_attribute(k,v)
            cam = self.world.spawn_actor(bp, transform, attach_to=attached)
            cam.listen(self.save_rgb_image)
            return cam
        elif st == 'LiDAR':
            bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            bp.set_attribute('range','100')
            bp.set_attribute('dropoff_general_rate',
                bp.get_attribute('dropoff_general_rate').recommended_values[0])
            bp.set_attribute('dropoff_intensity_limit',
                bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            bp.set_attribute('dropoff_zero_intensity',
                bp.get_attribute('dropoff_zero_intensity').recommended_values[0])
            for k,v in opts.items(): bp.set_attribute(k,v)
            l = self.world.spawn_actor(bp, transform, attach_to=attached)
            l.listen(self.save_lidar_image)
            return l
        elif st == 'SemanticLiDAR':
            bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            bp.set_attribute('range','100')
            for k,v in opts.items(): bp.set_attribute(k,v)
            l = self.world.spawn_actor(bp, transform, attach_to=attached)
            l.listen(self.save_sem_lidar)
            return l
        return None

    def save_rgb_image(self, image):
        image.convert(carla.ColorConverter.Raw)
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height,image.width,4))[:,:,:3][:,:,::-1]
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(arr.swapaxes(0,1))

    def _pts_to_surface(self, raw, cols, ds, rng):
        pts = np.frombuffer(raw, dtype=np.float32).reshape((-1,cols))[:,:2]
        pts = pts * (min(ds)/rng) + (0.5*ds[0], 0.5*ds[1])
        pts = np.fabs(pts).astype(np.int32)
        pts[:,0] = np.clip(pts[:,0], 0, ds[0]-1)
        pts[:,1] = np.clip(pts[:,1], 0, ds[1]-1)
        img = np.zeros((ds[0],ds[1],3), dtype=np.uint8)
        img[tuple(pts.T)] = (255,255,255)
        return pygame.surfarray.make_surface(img)

    def save_lidar_image(self, image):
        ds  = self.display_man.get_display_size()
        rng = 2.0 * float(self.sensor_options['range'])
        if self.display_man.render_enabled():
            self.surface = self._pts_to_surface(image.raw_data, 4, ds, rng)

    def save_sem_lidar(self, image):
        ds  = self.display_man.get_display_size()
        rng = 2.0 * float(self.sensor_options['range'])
        if self.display_man.render_enabled():
            self.surface = self._pts_to_surface(image.raw_data, 6, ds, rng)

    def render(self):
        if self.surface is not None:
            off = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, off)
            if self.label:
                lbl = self._font.render(self.label, True, (200,200,200))
                self.display_man.display.blit(lbl, (off[0]+6, off[1]+4))

    def destroy(self):
        if self.sensor: self.sensor.destroy()


# ══════════════════════════════════════════════════════════════════
# DetectionSensorManager - YOLOv8 + 주변액터 바운딩박스
# ══════════════════════════════════════════════════════════════════
class DetectionSensorManager(SensorManager):
    _CAM_YAW = {'FRONT': 0, 'LEFT': -90, 'RIGHT': 90}

    def __init__(self, world, display_man, transform, attached,
                 display_pos, label, ego_vehicle, safe_ctrl):
        self._raw_frame   = None
        self._det_surface = None
        self._frame_lock  = threading.Lock()
        self._det_lock    = threading.Lock()
        self._running     = True
        self._det_times   = collections.deque(maxlen=20)
        self._det_count   = 0
        self._det_fps     = 0.0
        self._ego         = ego_vehicle
        self._safe_ctrl   = safe_ctrl
        self._cam_yaw_off = self._CAM_YAW.get(label, 0)

        super().__init__(world, display_man, 'RGBCamera', transform,
                         attached, {}, display_pos, label=label)
        _get_yolo()
        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()

    def save_rgb_image(self, image):
        image.convert(carla.ColorConverter.Raw)
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        rgb = arr.reshape((image.height,image.width,4))[:,:,:3][:,:,::-1].copy()
        with self._frame_lock:
            self._raw_frame = rgb
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(rgb.swapaxes(0,1))

    def _project(self, actor, cam_tf, K, iw, ih):
        ext  = actor.bounding_box.extent
        v_tf = actor.get_transform()
        cns  = [carla.Location(sx*ext.x, sy*ext.y, sz*ext.z)
                for sx in (1,-1) for sy in (1,-1) for sz in (1,-1)]
        cl   = cam_tf.location
        cy   = math.radians(cam_tf.rotation.yaw)
        cp   = math.radians(cam_tf.rotation.pitch)
        xs, ys = [], []
        for lc in cns:
            wc = v_tf.transform(lc)
            dx,dy,dz = wc.x-cl.x, wc.y-cl.y, wc.z-cl.z
            ccy,scy  = math.cos(-cy), math.sin(-cy)
            rx =  ccy*dx + scy*dy
            ry = -scy*dx + ccy*dy
            rz =  dz
            ccp,scp  = math.cos(-cp), math.sin(-cp)
            fx =  ccp*rx - scp*rz
            fz =  scp*rx + ccp*rz
            fy =  ry
            cx2,cy2,cz2 = fy,-fz,fx
            if cz2 <= 0.1: continue
            xs.append(int(K[0,0]*cx2/cz2 + K[0,2]))
            ys.append(int(K[1,1]*cy2/cz2 + K[1,2]))
        if len(xs) < 2: return None
        x1=max(0,min(xs)); x2=min(iw-1,max(xs))
        y1=max(0,min(ys)); y2=min(ih-1,max(ys))
        return (x1,y1,x2,y2) if x2>x1 and y2>y1 else None

    def _detect_loop(self):
        import cv2
        model  = _get_yolo()
        ds     = self.display_man.get_display_size()
        iw, ih = ds[0], ds[1]
        f = iw / (2.0*math.tan(math.radians(45.0)))
        K = np.array([[f,0,iw/2],[0,f,ih/2],[0,0,1]], dtype=np.float64)

        while self._running:
            with self._frame_lock:
                frame = self._raw_frame
            if frame is None:
                time.sleep(0.03); continue

            # YOLO 추론 (차량/사람 클래스만, 신뢰도 0.4 이상)
            # COCO 클래스: 0=person, 1=bicycle, 2=car, 3=motorcycle,
            #              5=bus, 7=truck (신호등/기둥 등 제외)
            YOLO_CLASSES = [0, 1, 2, 3, 5, 7]
            t0 = time.time()
            with _yolo_lock:
                results = model(frame, verbose=False,
                                classes=YOLO_CLASSES, conf=0.4)[0]
            elapsed = time.time() - t0
            det_rgb = results.plot()[:,:,::-1].copy()

            # 주변 액터 바운딩박스 오버레이
            try:
                cam_tf   = self.sensor.get_transform()
                ego_tf   = self._ego.get_transform()
                ego_loc  = ego_tf.location
                ego_yaw  = ego_tf.rotation.yaw
                # 이 카메라의 절대 방향
                cam_abs_yaw = ego_yaw + self._cam_yaw_off

                # 감지 존에 있는 액터 ID
                danger_ids = set()
                for det in [self._safe_ctrl.det_front, self._safe_ctrl.det_rear,
                             self._safe_ctrl.det_left,  self._safe_ctrl.det_right]:
                    if det.actor:
                        danger_ids.add(det.actor.id)

                targets  = list(self.world.get_actors().filter('vehicle.*'))
                targets += list(self.world.get_actors().filter('walker.pedestrian.*'))

                for actor in targets:
                    if actor.id == self._ego.id: continue
                    aloc  = actor.get_location()
                    dx    = aloc.x - ego_loc.x
                    dy    = aloc.y - ego_loc.y
                    dist  = math.sqrt(dx*dx + dy*dy)
                    if dist > 50.0 or dist < 0.3: continue

                    # 이 카메라 방향 기준 ±65도 이내만 표시
                    rel = angle_diff(math.degrees(math.atan2(dy,dx)), cam_abs_yaw)
                    if abs(rel) > 65.0: continue

                    rect = self._project(actor, cam_tf, K, iw, ih)
                    if rect is None: continue
                    x1,y1,x2,y2 = rect

                    is_ped    = 'walker' in actor.type_id
                    is_danger = actor.id in danger_ids

                    if is_danger:
                        col, tk = (255,0,0), 3
                    elif is_ped:
                        col, tk = (255,100,255), 2
                    elif dist < 10:
                        col, tk = (255,80,80), 2
                    elif dist < 25:
                        col, tk = (255,200,0), 2
                    else:
                        col, tk = (60,220,60), 1

                    cv2.rectangle(det_rgb,(x1,y1),(x2,y2),col,tk)
                    name = 'pedestrian' if is_ped \
                           else actor.type_id.split('.')[-1][:10]
                    txt  = f"[!]{name} {dist:.1f}m" if is_danger \
                           else f"{name} {dist:.1f}m"
                    (tw,th),_ = cv2.getTextSize(txt,
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
                    by = max(0,y1-th-6)
                    cv2.rectangle(det_rgb,(x1,by),(x1+tw+6,y1),col,-1)
                    cv2.putText(det_rgb,txt,(x1+3,max(th,y1-3)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,0,0),1,cv2.LINE_AA)

            except Exception:
                pass

            self._det_times.append(elapsed)
            self._det_fps   = 1.0/(sum(self._det_times)/len(self._det_times))
            self._det_count = len(results.boxes)
            surf = pygame.surfarray.make_surface(det_rgb.swapaxes(0,1))
            with self._det_lock:
                self._det_surface = surf
            time.sleep(0.01)

    def render(self):
        with self._det_lock:
            det = self._det_surface
        surf = det if det is not None else self.surface
        if surf is None: return
        off    = self.display_man.get_display_offset(self.display_pos)
        ds     = self.display_man.get_display_size()
        active = det is not None
        self.display_man.display.blit(surf, off)
        col = (0,220,80) if active else (140,140,140)
        pygame.draw.rect(self.display_man.display, col,
                         (off[0],off[1],ds[0],ds[1]), 2)
        lbl = self._font.render(
            f'{self.label}  {"DET" if active else "..."}', True, col)
        self.display_man.display.blit(lbl,(off[0]+6,off[1]+4))
        if active:
            info = self._font.render(
                f'{self._det_count} obj  {self._det_fps:.1f}fps', True,(0,220,80))
            self.display_man.display.blit(info,(off[0]+6,off[1]+ds[1]-20))

    def destroy(self):
        self._running = False
        super().destroy()


# ══════════════════════════════════════════════════════════════════
# HUD 오버레이
# ══════════════════════════════════════════════════════════════════
class HUDOverlay:
    def __init__(self, display, win_w, win_h):
        self.display = display
        self.win_w   = win_w
        self.win_h   = win_h
        self.font_lg = pygame.font.SysFont('malgungothic', 22, bold=True)
        self.font_md = pygame.font.SysFont('malgungothic', 16)
        self.font_sm = pygame.font.SysFont('malgungothic', 13)
        self.collision_count = 0
        self.collision_flash = 0
        self.lane_msg        = ''
        self.lane_timer      = 0
        self.weather_idx     = 0
        self.weather_timer   = 0.0
        self.clock           = pygame.time.Clock()

    def on_collision(self, event):
        other = event.other_actor.type_id
        if any(other.startswith(p) for p in IGNORE_COLLISION):
            return
        self.collision_count += 1
        self.collision_flash  = 25
        print(f"[충돌] {other}  총 {self.collision_count}회")

    def on_lane_invasion(self, event):
        markings = list(set(str(m.type) for m in event.crossed_lane_markings))
        self.lane_msg   = '차선이탈: ' + ', '.join(markings)
        self.lane_timer = 60

    def update_weather(self, world, dt):
        self.weather_timer += dt
        if self.weather_timer >= WEATHER_INTERVAL:
            self.weather_timer = 0.0
            self.weather_idx   = (self.weather_idx+1) % len(WEATHER_CYCLE)
            name, preset = WEATHER_CYCLE[self.weather_idx]
            world.set_weather(preset)
            print(f"[날씨] → {name}")

    def _draw_radar(self, safe_ctrl):
        """360도 레이더 미니맵"""
        cw  = self.win_w // 3
        ch  = self.win_h // 2
        cx  = cw // 2
        cy  = ch + ch // 2
        r   = 52
        ox, oy = r+4, r+4
        sz  = r*2+8

        s = pygame.Surface((sz, sz), pygame.SRCALPHA)
        pygame.draw.circle(s,(0,0,0,150),(ox,oy),r)
        pygame.draw.circle(s,(60,60,60,180),(ox,oy),r,1)
        pygame.draw.circle(s,(50,50,50,80),(ox,oy),r//2,1)
        pygame.draw.line(s,(70,70,70,100),(ox,oy-r),(ox,oy+r),1)
        pygame.draw.line(s,(70,70,70,100),(ox-r,oy),(ox+r,oy),1)

        ego_tf  = safe_ctrl.vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_yaw = ego_tf.rotation.yaw

        # 감지 존 부채꼴 표시
        zone_cfg = {
            'FRONT': (0,   FRONT_HALF_ANGLE, SLOW_DIST,  (80,200,80,35)),
            'REAR':  (180, REAR_HALF_ANGLE,  REAR_DIST,  (200,80,80,35)),
            'LEFT':  (-90, SIDE_HALF_ANGLE,  SIDE_DIST,  (80,120,200,35)),
            'RIGHT': (90,  SIDE_HALF_ANGLE,  SIDE_DIST,  (80,120,200,35)),
        }
        scale = r / 25.0
        for zname,(base,half,zr,zcol) in zone_cfg.items():
            zr_px = int(zr*scale*0.9)
            steps = 12
            pts   = [(ox,oy)]
            for i in range(steps+1):
                a = math.radians(ego_yaw + base - half + (2*half)*i/steps)
                pts.append((int(ox+math.cos(a)*zr_px),
                             int(oy+math.sin(a)*zr_px)))
            if len(pts) >= 3:
                try: pygame.draw.polygon(s, zcol, pts)
                except: pass

        # 에고 차량 삼각형
        pygame.draw.polygon(s,(255,255,255),
            [(ox,oy-7),(ox-4,oy+5),(ox+4,oy+5)])

        # 감지된 액터
        seen = set()
        for det in [safe_ctrl.det_front, safe_ctrl.det_rear,
                    safe_ctrl.det_left,  safe_ctrl.det_right]:
            if det.actor and det.actor.id not in seen:
                seen.add(det.actor.id)
                aloc = det.actor.get_location()
                dx   = aloc.x - ego_loc.x
                dy   = aloc.y - ego_loc.y
                px   = int(ox + dx*scale)
                py   = int(oy + dy*scale)
                px   = max(4, min(sz-4, px))
                py   = max(4, min(sz-4, py))
                is_ped = 'walker' in det.actor.type_id
                col = (255,100,255,230) if is_ped else (255,60,60,230)
                pygame.draw.circle(s, col, (px,py), 4)

        self.display.blit(s,(cx-r-4, cy-r-4))
        lbl = self.font_sm.render("360° RADAR", True, (140,140,140))
        self.display.blit(lbl,(cx-lbl.get_width()//2, cy+r+2))

    def render(self, vehicle, safe_ctrl):
        speed  = get_speed_kmh(vehicle)
        fps    = self.clock.get_fps()
        w_name = WEATHER_CYCLE[self.weather_idx][0]
        cw     = self.win_w // 3
        ch     = self.win_h // 2

        # ── 속도계 (하단 중앙) ────────────────────────────────────
        ov = pygame.Surface((cw,100), pygame.SRCALPHA)
        ov.fill((0,0,0,170))
        self.display.blit(ov,(cw,ch))

        sc = (80,220,120) if speed<50 else (255,200,0) if speed<70 else (255,60,60)
        draw_text(self.display, f"{speed:.1f} km/h", (cw+10,ch+5), self.font_lg, color=sc)
        bw = int(min(speed/80.0,1.0)*(cw-20))
        pygame.draw.rect(self.display,(50,50,50),(cw+10,ch+38,cw-20,7))
        pygame.draw.rect(self.display,sc,         (cw+10,ch+38,bw,    7))

        st     = safe_ctrl.status
        st_col = (255,60,60)  if any(w in st for w in ['정지','재탐색']) \
                 else (255,200,0) if any(w in st for w in ['감속','후방','끼어']) \
                 else (60,220,120)
        draw_text(self.display, st,  (cw+10,ch+52), self.font_md, color=st_col)
        draw_text(self.display, f"FPS {fps:.1f}  |  {w_name}",
                  (cw+10,ch+76), self.font_sm, color=(180,180,180))

        # ── 방향별 감지 정보 (하단 우) ────────────────────────────
        det_all = safe_ctrl.all_detections()
        cov = pygame.Surface((cw, max(30, 10+20*len(det_all))), pygame.SRCALPHA)
        cov.fill((0,0,0,150))
        self.display.blit(cov,(cw*2,ch))
        if det_all:
            for i,(zn,det) in enumerate(det_all.items()):
                kind = '보행자' if 'walker' in det.actor.type_id else '차량'
                col  = (255,80,80)  if zn=='FRONT' \
                       else (255,160,0) if zn=='REAR' \
                       else (100,180,255)
                draw_text(self.display,
                          f"[{zn}] {kind} {det.dist:.1f}m",
                          (cw*2+8,ch+4+i*20), self.font_sm, color=col)
        else:
            ccol = (255,80,80) if self.collision_count>0 else (180,180,180)
            draw_text(self.display,
                      f"충돌 {self.collision_count}회  NPC차15 보행자10",
                      (cw*2+8,ch+6), self.font_sm, color=ccol)

        # 날씨 전환 타이머
        draw_text(self.display, f"날씨전환 {WEATHER_INTERVAL-self.weather_timer:.0f}s",
                  (self.win_w-130,8), self.font_sm, color=(180,180,255))

        # 360도 레이더
        self._draw_radar(safe_ctrl)

        # ── 충돌 플래시 ───────────────────────────────────────────
        if self.collision_flash > 0:
            alpha = int(160*self.collision_flash/25)
            fl = pygame.Surface((self.win_w,self.win_h), pygame.SRCALPHA)
            fl.fill((255,0,0,alpha))
            self.display.blit(fl,(0,0))
            draw_text(self.display, f"COLLISION! ({self.collision_count})",
                      (self.win_w//2-130,self.win_h//2-18),
                      self.font_lg, color=(255,255,255))
            self.collision_flash -= 1

        # ── 상태 배너 ─────────────────────────────────────────────
        banner_cfg = {
            '정지':  ((200,0,0,180),   f"[긴급 정지]  전방 {safe_ctrl.det_front.dist:.1f}m"),
            '끼어':  ((0,100,180,160),  "[측면 끼어들기 감지!]  속도 조절 중"),
            '후방':  ((150,80,0,160),   f"[후방 접근]  {safe_ctrl.det_rear.dist:.1f}m"),
            '감속':  ((180,130,0,150),  f"[감속]  전방 {safe_ctrl.det_front.dist:.1f}m"),
        }
        for key,(col,msg) in banner_cfg.items():
            if key in st:
                so = pygame.Surface((self.win_w,40), pygame.SRCALPHA)
                so.fill(col)
                self.display.blit(so,(0,ch-40))
                draw_text(self.display, msg,
                          (self.win_w//2-len(msg)*5, ch-32),
                          self.font_md, color=(255,255,255))
                break

        # ── 차선이탈 경고 ─────────────────────────────────────────
        if self.lane_timer > 0:
            wo = pygame.Surface((500,32), pygame.SRCALPHA)
            wo.fill((255,200,0,100))
            wx = self.win_w//2-250
            self.display.blit(wo,(wx,ch-38))
            draw_text(self.display, f"⚠  {self.lane_msg}",
                      (wx+10,ch-32), self.font_md, color=(255,230,0))
            self.lane_timer -= 1

        self.clock.tick(30)


# ══════════════════════════════════════════════════════════════════
# 메인 시뮬레이션
# ══════════════════════════════════════════════════════════════════
def run_simulation(args, client):
    display_manager  = None
    vehicle_list     = []
    walker_list      = []
    walker_ctrl_list = []
    extra_sensors    = []
    hud              = None
    safe_ctrl        = None
    timer            = CustomTimer()

    try:
        world    = client.get_world()
        bp_lib   = world.get_blueprint_library()
        orig_cfg = world.get_settings()
        tm       = client.get_trafficmanager(8000)

        if args.sync:
            s = world.get_settings()
            tm.set_synchronous_mode(True)
            s.synchronous_mode    = True
            s.fixed_delta_seconds = 0.05
            world.apply_settings(s)

        spawn_pts = world.get_map().get_spawn_points()

        # 에고 차량
        ego_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        ego_bp.set_attribute('role_name', 'hero')
        ego = world.spawn_actor(ego_bp, random.choice(spawn_pts))
        vehicle_list.append(ego)
        print(f"[+] 에고 차량: {ego_bp.id}")

        safe_ctrl = SafeAutopilot(ego, world, tm)

        # NPC 차량 15대
        npc_bps = bp_lib.filter('vehicle.*')
        spawned = 0
        for sp in random.sample(spawn_pts, min(50,len(spawn_pts))):
            if spawned >= 15: break
            bp = random.choice(npc_bps)
            if bp.has_attribute('color'):
                bp.set_attribute('color',
                    random.choice(bp.get_attribute('color').recommended_values))
            npc = world.try_spawn_actor(bp, sp)
            if npc:
                npc.set_autopilot(True, tm.get_port())
                tm.auto_lane_change(npc, True)   # NPC는 차선변경 허용 (자연스러운 교통)
                vehicle_list.append(npc)
                spawned += 1
        print(f"[+] NPC 차량 {spawned}대")

        # 보행자 10명
        walker_bps = bp_lib.filter('walker.pedestrian.*')
        w_locs = []
        for _ in range(20):
            loc = world.get_random_location_from_navigation()
            if loc: w_locs.append(carla.Transform(loc))

        batch = []
        for sp in w_locs[:10]:
            wbp = random.choice(walker_bps)
            if wbp.has_attribute('is_invincible'):
                wbp.set_attribute('is_invincible','false')
            batch.append(carla.command.SpawnActor(wbp, sp))
        for res in client.apply_batch_sync(batch, True):
            if not res.error:
                walker_list.append(world.get_actor(res.actor_id))

        ctrl_bp = bp_lib.find('controller.ai.walker')
        batch2  = [carla.command.SpawnActor(ctrl_bp, carla.Transform(), w)
                   for w in walker_list]
        for res in client.apply_batch_sync(batch2, True):
            if not res.error:
                c = world.get_actor(res.actor_id)
                walker_ctrl_list.append(c)
                c.start()
                c.go_to_location(world.get_random_location_from_navigation())
                c.set_max_speed(1.0 + random.random())
        print(f"[+] 보행자 {len(walker_list)}명")

        # DisplayManager
        display_manager = DisplayManager(
            grid_size=[2,3], window_size=[args.width, args.height])
        hud = HUDOverlay(display_manager.display, args.width, args.height)
        world.set_weather(WEATHER_CYCLE[0][1])

        # 충돌 / 차선이탈 센서
        col_sen = world.spawn_actor(
            bp_lib.find('sensor.other.collision'),
            carla.Transform(), attach_to=ego)
        col_sen.listen(hud.on_collision)
        extra_sensors.append(col_sen)

        lane_sen = world.spawn_actor(
            bp_lib.find('sensor.other.lane_invasion'),
            carla.Transform(), attach_to=ego)
        lane_sen.listen(hud.on_lane_invasion)
        extra_sensors.append(lane_sen)

        # 상단 YOLO 카메라 3개
        for yaw,pos,lbl in [(-90,[0,0],'LEFT'),(0,[0,1],'FRONT'),(90,[0,2],'RIGHT')]:
            DetectionSensorManager(
                world, display_manager,
                carla.Transform(carla.Location(x=0,z=2.4), carla.Rotation(yaw=yaw)),
                ego, display_pos=pos, label=lbl,
                ego_vehicle=ego, safe_ctrl=safe_ctrl)

        # 하단 센서
        SensorManager(world, display_manager, 'LiDAR',
            carla.Transform(carla.Location(x=0,z=2.4)), ego,
            {'channels':'64','range':'100',
             'points_per_second':'250000','rotation_frequency':'20'},
            display_pos=[1,0], label='LiDAR')

        SensorManager(world, display_manager, 'RGBCamera',
            carla.Transform(carla.Location(x=0,z=2.4), carla.Rotation(yaw=180)),
            ego, {}, display_pos=[1,1], label='REAR')

        SensorManager(world, display_manager, 'SemanticLiDAR',
            carla.Transform(carla.Location(x=0,z=2.4)), ego,
            {'channels':'64','range':'100',
             'points_per_second':'100000','rotation_frequency':'20'},
            display_pos=[1,2], label='Semantic LiDAR')

        print("\n[*] 시뮬레이션 시작! ESC/Q 종료")
        print("[*] YOLO 로딩 중 (10~20초)...")
        print("[*] 전방: 같은차선만 감지 | 측면: 실제 끼어들기만 감지\n")

        last_t = timer.time()
        call_exit = False

        while not call_exit:
            now = timer.time()
            dt  = now - last_t
            last_t = now

            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            safe_ctrl.run_step(dt=dt)
            display_manager.render()
            hud.update_weather(world, dt)
            hud.render(ego, safe_ctrl)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key in (K_ESCAPE, K_q):
                        call_exit = True

    finally:
        print("\n[*] 정리 중...")
        for c in walker_ctrl_list:
            try: c.stop(); c.destroy()
            except: pass
        for s in extra_sensors:
            try: s.destroy()
            except: pass
        if display_manager:
            display_manager.destroy()
        client.apply_batch(
            [carla.command.DestroyActor(x)
             for x in vehicle_list + walker_list])
        try: world.apply_settings(orig_cfg)
        except: pass
        pygame.quit()
        if hud:
            print(f"[+] 종료 | 총 충돌 {hud.collision_count}회")


# ══════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description='Capstone CARLA Final Demo')
    ap.add_argument('--host',  default='127.0.0.1')
    ap.add_argument('-p','--port', default=2000, type=int)
    ap.add_argument('--sync',  action='store_true', default=True)
    ap.add_argument('--async', dest='sync', action='store_false')
    ap.add_argument('--res',   default='1280x720')
    args = ap.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    print("=" * 56)
    print("  Capstone Design - CARLA 0.9.16 Final Demo")
    print("  Autopilot + 360도감지 + YOLO + 멀티센서")
    print("=" * 56 + "\n")

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        run_simulation(args, client)
    except KeyboardInterrupt:
        print('\n[!] 사용자 종료')
    except Exception as e:
        print(f'\n[ERROR] {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
