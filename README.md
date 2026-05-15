# 🚗 Capstone Design — CARLA 0.9.16 자율주행 시뮬레이터

> CARLA 시뮬레이터 기반 자율주행 + 실시간 멀티센서 퓨전 + YOLOv8 객체 인식 데모

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 시뮬레이터 | CARLA 0.9.16 (Unreal Engine 4 기반) |
| 언어 | Python 3.12 |
| 핵심 라이브러리 | carla, pygame, numpy, ultralytics (YOLOv8), opencv |
| 자율주행 방식 | CARLA Traffic Manager Autopilot |
| 객체 인식 | YOLOv8n (사람·차량 클래스 한정) |
| NPC | 차량 15대 + 보행자 10명 |

---

## 🗺️ 전체 시스템 구조

```mermaid
flowchart TD
    A[CarlaUE4.exe\n시뮬레이터 서버] -->|TCP 2000| B[Python 클라이언트\ncarla_project1_final.py]

    B --> C[씬 구성]
    C --> C1[에고 차량 스폰\nTesla Model3]
    C --> C2[NPC 차량 15대]
    C --> C3[보행자 10명\nAI Walker]

    B --> D[센서 부착]
    D --> D1[전방 카메라\nYOLOv8 DET]
    D --> D2[좌측 카메라\nYOLOv8 DET]
    D --> D3[우측 카메라\nYOLOv8 DET]
    D --> D4[후방 카메라\nRGB]
    D --> D5[LiDAR\n64ch / 100m]
    D --> D6[Semantic LiDAR\n64ch / 100m]
    D --> D7[충돌 센서]
    D --> D8[차선이탈 센서]

    B --> E[SafeAutopilot\n360도 감지 컨트롤러]
    B --> F[HUD 오버레이\npygame 렌더링]
    B --> G[동적 날씨\n12초 자동 전환]
```

---

## 🔄 메인 루프 동작 순서

```mermaid
sequenceDiagram
    participant S as CARLA 서버
    participant A as SafeAutopilot
    participant R as 렌더러(pygame)
    participant H as HUD

    loop 매 프레임 (30fps)
        S->>A: world.tick()
        A->>A: 360도 스캔 (전/후/좌/우)
        A->>A: 상태 판단 및 제어
        A->>S: apply_control() or autopilot 속도조절
        S->>R: 센서 데이터 전달
        R->>R: 카메라 / LiDAR 렌더링
        R->>R: YOLO 추론 (별도 스레드)
        R->>H: 화면 합성 (pygame.display.flip)
    end
```

---

## 🧠 SafeAutopilot 상태 머신

```mermaid
stateDiagram-v2
    [*] --> 자율주행중

    자율주행중 --> 전방감속중 : 전방 동일차선 20m 이내
    자율주행중 --> 전방긴급정지 : 전방 동일차선 8m 이내
    자율주행중 --> 끼어들기감지 : 측면 실제 끼어들기 감지
    자율주행중 --> 후방차량접근 : 후방 동일차선 12m 이내

    전방감속중 --> 자율주행중 : 장애물 소멸
    전방감속중 --> 전방긴급정지 : 8m 이내 진입

    전방긴급정지 --> 자율주행중 : 장애물 소멸

    끼어들기감지 --> 자율주행중 : 끼어들기 완료
    후방차량접근 --> 자율주행중 : 거리 확보
```

---

## 📡 360도 감지 존

```mermaid
flowchart LR
    subgraph 감지존["360° 감지 존"]
        direction TB
        F["🔴 FRONT\n±20도 / 20m\n감속·긴급정지"]
        L["🔵 LEFT\n±10도 / 6m\n끼어들기 감지"]
        R["🔵 RIGHT\n±10도 / 6m\n끼어들기 감지"]
        B["🟠 REAR\n±20도 / 12m\n후방 접근 경고"]
    end

    subgraph 판단["끼어들기 판별 (3조건 중 2개)"]
        M1["① 횡방향 거리 < 3.5m"]
        M2["② 에고 근방 위치\n-5m ~ +10m"]
        M3["③ 속도벡터가 에고 방향"]
    end

    L --> 판단
    R --> 판단
```

---

## 🖥️ 화면 레이아웃

```mermaid
block-beta
    columns 3
    A["📷 LEFT\nYOLOv8 DET"]:1
    B["📷 FRONT\nYOLOv8 DET"]:1
    C["📷 RIGHT\nYOLOv8 DET"]:1
    D["📡 LiDAR\n탑뷰 + 레이더"]:1
    E["📷 REAR\n후방 카메라"]:1
    F["🔬 Semantic\nLiDAR"]:1
```

**HUD 정보 (하단 오버레이)**
- 속도계 (km/h) + 게이지 바
- 주행 상태 텍스트 (감속 / 정지 / 끼어들기 등)
- 360° 레이더 미니맵
- FPS / 날씨 / 충돌 횟수 / 방향별 감지 거리
- 충돌 시 전체화면 빨간 플래시
- 차선이탈 경고 배너

---

## 🎨 바운딩박스 색상 기준

| 색상 | 의미 |
|---|---|
| 🔴 빨강 (굵은 테두리) | 위험 감지 존 액터 |
| 🟣 보라 | 보행자 |
| 🟠 주황 | 10m 이내 근접 차량 |
| 🟡 노랑 | 10~25m 주의 차량 |
| 🟢 초록 | 25m 이상 안전 거리 |

---



## 🌦️ 동적 날씨 사이클

```mermaid
flowchart LR
    W1[☀️ Clear Noon] -->|12초| W2[🌥️ Cloudy]
    W2 -->|12초| W3[🌧️ Wet]
    W3 -->|12초| W4[🌦️ Soft Rain]
    W4 -->|12초| W5[⛈️ Hard Rain]
    W5 -->|12초| W6[🌫️ Foggy]
    W6 -->|12초| W1
```

---

## 📦 주요 클래스 구성

```mermaid
classDiagram
    class SafeAutopilot {
        +status : str
        +det_front / rear / left / right
        +run_step(dt)
        -_scan()
        -_is_same_lane()
        -_is_merging()
    }

    class DetectionSensorManager {
        +label : str
        -_detect_loop() Thread
        -_project() 3D→2D
        +render()
    }

    class SensorManager {
        +sensor_type
        +save_rgb_image()
        +save_lidar_image()
        +save_sem_lidar()
        +render()
    }

    class HUDOverlay {
        +collision_count
        +weather_idx
        +render(vehicle, safe_ctrl)
        -_draw_radar()
        +on_collision()
        +on_lane_invasion()
    }

    class DisplayManager {
        +grid_size [2x3]
        +sensor_list
        +render()
    }

    DetectionSensorManager --|> SensorManager
    DisplayManager "1" o-- "6" SensorManager
    HUDOverlay --> SafeAutopilot
```
