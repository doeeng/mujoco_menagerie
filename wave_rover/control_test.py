import mujoco
import mujoco.viewer
import numpy as np
import threading
from pynput import keyboard
import cv2

# MuJoCo 모델 불러오기
model = mujoco.MjModel.from_xml_path(
    "/home/oh/my_mujoco/mujoco_menagerie/wave_rover/scene_road.xml"
)
data = mujoco.MjData(model)

# 렌더링용 context (offscreen)
offscreen = mujoco.Renderer(model)

# 카메라 이름 찾기
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
if camera_id == -1:
    raise RuntimeError("카메라 'cam'을 찾을 수 없습니다.")

# 키보드 상태 변수
velocity = 0.0
turn = 0.0

# 키 상태를 저장하는 변수
keys_pressed = set()


# 키 입력 업데이트 함수
def on_press(key):
    keys_pressed.add(key)


def on_release(key):
    keys_pressed.discard(key)
    # 프로그램 종료를 원하면 Esc 키로
    if key == keyboard.Key.esc:
        return False


# 키 입력 처리 루프 (별도 쓰레드)
def keyboard_control():
    global velocity, turn
    while True:
        if keyboard.Key.up in keys_pressed:
            velocity += 0.2
        if keyboard.Key.down in keys_pressed:
            velocity -= 0.2
        if keyboard.Key.left in keys_pressed:
            turn -= 0.2
        if keyboard.Key.right in keys_pressed:
            turn += 0.2
        if keyboard.Key.space in keys_pressed:
            velocity = 0.0
            turn = 0.0

        # 속도 제한
        velocity = np.clip(velocity, -10.0, 10.0)
        turn = np.clip(turn, -5.0, 5.0)

        # 100Hz 루프 대기
        threading.Event().wait(0.01)


# 키보드 리스너 시작
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# 키 입력 처리 쓰레드 시작
threading.Thread(target=keyboard_control, daemon=True).start()

# OpenCV 창 준비
cv2.namedWindow("MuJoCo Camera View", cv2.WINDOW_NORMAL)

# MuJoCo Viewer 실행
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = viewer.sync()

        # 왼쪽/오른쪽 바퀴 속도
        left = velocity + turn
        right = velocity - turn
        data.ctrl[:] = [left, right, left, right]

        mujoco.mj_step(model, data)

        # 오프스크린 렌더링
        offscreen.update_scene(data, camera=camera_id)
        pixels = offscreen.render()

        # OpenCV에 표시 (RGB -> BGR 변환)
        pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        cv2.imshow("MuJoCo Camera View", pixels_bgr)

        # ESC 키 누르면 종료
        if cv2.waitKey(1) == 27:
            break

# OpenCV 창 닫기
cv2.destroyAllWindows()
