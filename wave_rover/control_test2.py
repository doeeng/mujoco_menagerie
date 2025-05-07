import mujoco
import mujoco.viewer
import numpy as np
import threading
from pynput import keyboard
import torch
from torchvision import transforms
import torchvision
import cv2
import PIL
import copy

# --- (1) mujoco 세팅 --- #
# MuJoCo 모델 로드
model_path = "/home/oh/my_mujoco/mujoco_menagerie/wave_rover/scene_road.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 오프스크린 렌더링 준비
offscreen = mujoco.Renderer(model, height=480, width=640)


# 카메라 설정
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
if camera_id == -1:
    raise RuntimeError("카메라 'cam'을 찾을 수 없습니다.")


# --- (2) 신경망 모델 로드 --- #
def get_model():
    model = torchvision.models.alexnet(num_classes=2, dropout=0.0)
    return model


device = torch.device("cpu")
policy = get_model()
policy.load_state_dict(
    torch.load(
        "/home/oh/my_mujoco/mujoco_menagerie/wave_rover/road_following_model.pth",
        map_location=device,
    )
)
policy = policy.to(device)
policy.eval()

# 이미지 전처리 파이프라인
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def preprocess(img):
    img = transform(img)
    return img.unsqueeze(0).to(device)  # (B, C, H, W)


# --- (3) 키보드 수동 조작 --- #
velocity = 0.0
turn = 0.0
keys_pressed = set()
use_policy = False  # "q"


def on_press(key):
    global use_policy
    keys_pressed.add(key)
    if hasattr(key, "char") and key.char == "q":
        print("[INFO] 전환: 모델 제어 모드로 진입합니다.")
        use_policy = True


def on_release(key):
    keys_pressed.discard(key)
    if key == keyboard.Key.esc:
        return False


def keyboard_control():
    global velocity, turn
    while True:
        if not use_policy:
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

            velocity = np.clip(velocity, -10.0, 10.0)
            turn = np.clip(turn, -5.0, 5.0)

        threading.Event().wait(0.01)


# --- (4) 키보드 리스너 및 쓰레드 시작 --- #

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
threading.Thread(target=keyboard_control, daemon=True).start()

# --- (5) 메인 루프 --- #

# OpenCV 창 설정
cv2.namedWindow("MuJoCo Camera View", cv2.WINDOW_NORMAL)

# MuJoCo Viewer 루프
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = viewer.sync()

        if use_policy:
            # 카메라 이미지 획득
            offscreen.update_scene(data, camera=camera_id)
            pixels = offscreen.render()
            img_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

            # PIL 이미지 변환
            img_pil = PIL.Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            width, height = img_pil.width, img_pil.height

            # 이미지 전처리 및 추론
            with torch.no_grad():
                input_tensor = preprocess(img_pil)
                output = policy(input_tensor).detach().cpu().numpy()

            # 예측 결과 변환
            x_norm, y_norm = output[0]  # [-1, 1] 범위
            x_pred = (x_norm / 2 + 0.5) * width
            y_pred = (y_norm / 2 + 0.5) * height

            # 조향 제어
            center_x = width / 2
            error = x_pred - center_x

            k_p = 0.01
            turn = -k_p * error  # 좌우 회전 방향 조정
            velocity = 5.0  # 일정한 전진 속도

        # 제어
        left = velocity + turn
        right = velocity - turn
        data.ctrl[:] = [left, right, left, right]

        mujoco.mj_step(model, data)

        if use_policy:
            # 화면 표시
            img_disp = copy.deepcopy(img_bgr)
            cv2.circle(
                img_disp, (int(x_pred), int(y_pred)), 5, (0, 255, 0), -1
            )  # Predicted point
            cv2.line(
                img_disp, (int(center_x), 0), (int(center_x), height), (0, 0, 255), 2
            )  # Center line
            cv2.imshow("MuJoCo Camera View", img_disp)

        else:
            # 화면 렌더링
            offscreen.update_scene(data, camera=camera_id)
            pixels = offscreen.render()
            img_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            cv2.imshow("MuJoCo Camera View", img_bgr)

        if cv2.waitKey(1) == 27:  # ESC
            break

cv2.destroyAllWindows()
