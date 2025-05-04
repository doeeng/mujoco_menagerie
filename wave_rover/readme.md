환경

- ubuntu 22.04
    - login 할 때, 화면 아래쪽 톱니바퀴 클릭 → Xorg 모드로 변경하여 로그인
    - mujoco gui 실행 가능.
- mujoco 3.3.1 버전

### mujoco 설치

<aside>
💡

다음의 git repo 에서 mujoco 최신판 다운로드:
https://github.com/google-deepmind/mujoco/releases

아마 대부분 이거일 듯 : 

[mujoco-3.3.2-linux-x86_64.tar.gz](https://github.com/google-deepmind/mujoco/releases/download/3.3.2/mujoco-3.3.2-linux-x86_64.tar.gz)

</aside>

### 다운로드 디렉토리 이동 → 압축해제 (터미널 말고 폴더로 직접 가서 압축해제)
### mujoco simulation 실행 정상 작동 확인하기
1. 먼저 압축해제된 폴더 이름을 mujoco 이렇게 바꾸는게 편리함.
2. 아래 커맨드는 압축해제 폴더 이름을 “mujoco-3.3.2~~~~” → “mujoco” 변경한 상황을 가정

```bash
cd ~/Downloads/mujoco/bin
```

```bash
./simulate
```

뭔가 시뮬레이션 같은 화면이 뜨면 정상.

### git 레포 다운로드 하기.
1. 터미널 새로 하나 열고 다음 입력

```bash
mkdir embed_project && cd embed_project
```

```bash
git clone https://github.com/donghoon11/mujoco_menagerie.git
```

### 실행해야할 파일 찾기
1. 여러가지 모델 중 우리가 쓰는 모델은 waves_rover
2. assets, meshes 는 로봇 모델링 파일
3. control_test.py 가 키보드를 이용해서 조작할 수 있는 파일.

### 다음으로 의존성 라이브러리 설치
        
```bash
pip install mujoco pynput
```
1. 대부분 numpy, threading, cv2 는 설치되었을 것으로 예상. (에러뜨면 pip install 로 해당 라이브러리 설치ㄱㄱ)

### 마지막으로 control_test.py 실행
1. 해당 파일 열어서 실행하거나,
2. 다음의 커맨드로 실행
```bash
python3 control_test.py
```