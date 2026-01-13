import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


# -------- GPU SSIM 计算 --------
def ssim_torch(img1, img2, window_size=11):
    """
    img1, img2: torch tensors, shape (1, 1, H, W), 0~1
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    gaussian = cv2.getGaussianKernel(window_size, 1.5)
    gaussian = torch.tensor(gaussian @ gaussian.T, device=img1.device).float()
    gaussian = gaussian.expand(1, 1, window_size, window_size)

    mu1 = F.conv2d(img1, gaussian, padding=window_size//2)
    mu2 = F.conv2d(img2, gaussian, padding=window_size//2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, gaussian, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, gaussian, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, gaussian, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()



# ---------- 主流程 ----------
def extract_keyframes_gpu(
    video_path,
    output_dir,
    ssim_thresh=0.90,
    min_interval=3,
    save_fps=1,
    scene_threshold=27.0
):

    os.makedirs(output_dir, exist_ok=True)

    # ---- PySceneDetect 检测镜头 ----
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=scene_threshold))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scenes = scene_manager.get_scene_list()

    print(f"🟦 检测到镜头数量: {len(scenes)}")

    scene_bounds = [(s[0].get_frames(), s[1].get_frames()) for s in scenes]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    force_interval = max(1, int(fps / save_fps))

    saved_count = 0
    last_save = -99999
    scene_idx = 0
    prev_tensor = None

    frame_idx = 0
    device = torch.device("cuda")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 镜头切换
        if scene_idx < len(scene_bounds):
            s, e = scene_bounds[scene_idx]
            if frame_idx == s:
                prev_tensor = None
            if frame_idx > e:
                scene_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = torch.tensor(gray, device=device).float() / 255.0
        img = img[None, None, :, :]

        save = False

        # 最低采样率
        if frame_idx - last_save >= force_interval:
            save = True

        # GPU SSIM（关键帧检测）
        if prev_tensor is not None:
            score = ssim_torch(prev_tensor, img)
            if score < ssim_thresh and frame_idx - last_save >= min_interval:
                save = True

        if save:
            # ⬇⬇⬇ 保存为原始视频帧号名称（6 位数字） ⬇⬇⬇
            out_path = os.path.join(output_dir, f"{frame_idx:06d}.png")
            cv2.imwrite(out_path, frame)
            saved_count += 1
            last_save = frame_idx

        prev_tensor = img
        frame_idx += 1

    cap.release()
    print(f"🎉 完成！共保存 {saved_count} 个关键帧（GPU 加速版）.")

if __name__ == "__main__":
    extract_keyframes_gpu(
        video_path="hw.mp4",
        output_dir="hwkfg_24",
        ssim_thresh=0.95,   # 越低越敏感 → 越密集
        min_interval=2,     # 越小 → 越密集
        save_fps=0.5,         # 每秒至少 1 帧（可改成0.5、2等）
        scene_threshold=27  # PySceneDetect 的内容检测阈值
    )

3
0.95
2
0.5
27
1607
0.72-0.76
