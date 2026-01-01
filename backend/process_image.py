import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from rain_harmonizer.inference import Harmonizer
from typing import Tuple, List
from PIL import Image
import tqdm
import replicate
from Background_Prompt import get_background_prompt
import torchvision
import tempfile
import os

class SelfieSegmentor:
    def __init__(self, model_path: str):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=False, output_confidence_masks=True)
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def segment(self, img_bgr):
        """
        output: float mask (H, W), [0,1]
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        result = self.segmenter.segment(mp_image)

        mask = result.confidence_masks[0].numpy_view()
        return mask
def make_edge_soft_alpha(mask, feather = 10):
    """
    construct soft alpha with only edge smoothing
    """

    # hard mask
    hard = (mask > 0.5).astype(np.uint8)

    # dist_in:  distance from foreground to boundary
    # dist_out: distance from background to boundary
    dist_in = cv2.distanceTransform(hard, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - hard, cv2.DIST_L2, 5)

    alpha = np.zeros_like(dist_in, dtype=np.float32)

    alpha[dist_in >= feather] = 1.0

    # edge: linear transition
    edge = (dist_in < feather) & (dist_out < feather)
    alpha[edge] = dist_in[edge] / feather

    return alpha
def to_binary_mask(mask, thresh=0.5):
    return (mask > thresh).astype(np.uint8)

def find_foot_xy(hard_mask, band_height = 20):
    """
    estimate the foot position from the hard mask
    """

    ys, xs = np.where(hard_mask > 0)
    if len(ys) == 0:
        raise ValueError("empty mask")

    foot_y = int(ys.max())

    # foot band
    y0 = max(0, foot_y - band_height)
    band = hard_mask[y0:foot_y + 1, :]

    ys_band, xs_band = np.where(band > 0)
    if len(xs_band) == 0:
        return int(np.median(xs)), foot_y

    foot_x = int(np.median(xs_band))

    return foot_x, foot_y
def estimate_person_bbox_from_foot(hard_mask, foot_x, foot_y):
    """
    from foot_y, estimate the bbox upward
    """
    best_y0 = None
    for h in range(100, foot_y, 4):
        score = hard_mask[foot_y -h-4:foot_y - h, foot_x -100:foot_x + 100].sum()
        if score < 60:
            best_y0 = foot_y - h
            break

    region = hard_mask[best_y0:foot_y, :]
    ys, xs = np.where(region > 0)
    x0 = int(xs.min())
    x1 = int(xs.max())

    return x0, best_y0, x1, foot_y

def segment_person_with_edge_alpha(img, segmentor, feather = 10):
    # original segmentation
    mask = segmentor.segment(img)
    if mask.ndim == 3:
        mask = mask.squeeze(-1)

    mask = mask.astype(np.float32)

    # construct edge-only alpha
    alpha = make_edge_soft_alpha(mask, feather=feather)

    alpha_3 = alpha[..., None]

    # split
    img_f = img.astype(np.float32)

    person = img_f * (1 - alpha_3)
    environment = img_f * alpha_3

    person = np.clip(person, 0, 255).astype(np.uint8)
    environment = np.clip(environment, 0, 255).astype(np.uint8)

    hard = (mask < 0.5).astype(np.uint8)
    foot_x, foot_y = find_foot_xy(hard)
    bbox = estimate_person_bbox_from_foot(hard, foot_x, foot_y)
    print(bbox)
    return person, environment, alpha, foot_y, bbox

def make_bg_blur_video(person, background, num_frames = 50, max_blur = 41): # 120

    assert person.shape == background.shape
    H, W, _ = person.shape
    if max_blur % 2 == 0:
        max_blur += 1

    def heavy_box_blur(img, rounds=6, ksize=31):
        out = img.copy()
        for _ in range(rounds):
            out = cv2.blur(out, (ksize, ksize))
        return out
    bg_blur_max = heavy_box_blur(background, rounds=6, ksize=31)
    cv2.imwrite("bg_blur_max.jpg", bg_blur_max)
    frames = []
    for i in tqdm.tqdm(range(num_frames)):
        t = i / (num_frames - 1)
        bg = background.astype(np.float32) * (1 - t) + bg_blur_max.astype(np.float32) * t
        frame = person.astype(np.float32) + bg
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        if (i == num_frames - 1):
            for j in range(30):
                frames.append(frame)
                # writer.write(frame)
        else:
            frames.append(frame)
            # writer.write(frame)
    bg_blur_max_2 = bg_blur_max.astype(np.float32) + person.astype(np.float32)
    bg_blur_max_2 = np.clip(bg_blur_max_2, 0, 255).astype(np.uint8)
    # writer.release()
    return frames, bg_blur_max_2

def draw_bbox(img, bbox, color=(0, 0, 255), thickness=3):
    x0, y0, x1, y1 = bbox
    vis = img.copy()
    cv2.rectangle(vis, (x0, y0), (x1, y1), color, thickness)
    return vis
def aspect_to_str(aspect):
    if abs(aspect - 4/3) < 0.05:
        return "4:3"
    elif abs(aspect - 3/4) < 0.05:
        return "3:4"
    elif abs(aspect - 16/9) < 0.05:
        return "16:9"
    elif abs(aspect - 9/16) < 0.05:
        return "9:16"
    else:
        return "1:1"
def gen_background(gen = True, aspect = 4/3, beast_id = 'lei'):
    if gen == False:
        with open("bg_with_person.jpg", "rb") as f:
            return f.read()
    
    prompt = get_background_prompt(beast_id)
    input = { "prompt": prompt, "aspect_ratio": aspect_to_str(float(aspect))}
    output = replicate.run("bytedance/seedream-4", input=input)

    with open("bg_with_person.jpg", "wb") as f:
        f.write(output[0].read())
    return output[0].read()

def modify_image(img, gen = True, aspect = 4/3):
    if gen == False:
        with open("bg_without_person.jpg", "rb") as f:
            return f.read()
    prompt = """
    remove the person from the image while keeping the background the same
    """
    input = {
        "prompt": prompt,
        "image_input": [open("bg_with_person.jpg", "rb")],
        "aspect_ratio": aspect_to_str(float(aspect))
    }

    output = replicate.run(
        "bytedance/seedream-4",
        input=input
    )

    with open("bg_without_person.jpg", "wb") as f:
        f.write(output[0].read())
    return output[0].read()

def ease_in_quad(t: float) -> float:
    return t * t

def ease_out_quad(t: float) -> float:
    return 1.0 - (1.0 - t) * (1.0 - t)

def ease_in_out_smooth(t: float) -> float:
    # smoothstep
    return t * t * (3 - 2 * t)


def clamp_int(v, lo, hi):
    return max(lo, min(hi, v))

def bbox_center(b):
    x0, y0, x1, y1 = b
    return (x0 + x1) * 0.5, (y0 + y1) * 0.5

def bbox_size(b):
    x0, y0, x1, y1 = b
    w = (x1 - x0 + 1)
    h = (y1 - y0 + 1)
    return float(w), float(h)

def crop_with_pad(img, x0, y0, x1, y1, pad):
    H, W = img.shape[:2]
    cx0 = clamp_int(x0 - pad, 0, W - 1)
    cy0 = clamp_int(y0 - pad, 0, H - 1)
    cx1 = clamp_int(x1 + pad, 0, W - 1)
    cy1 = clamp_int(y1 + pad, 0, H - 1)
    return img[cy0:cy1+1, cx0:cx1+1].copy(), (cx0, cy0, cx1, cy1)

def very_blurry(img, downsample = 16, extra_gauss = 31):
    H, W = img.shape[:2]
    ds = max(2, int(downsample))
    h2 = max(2, H // ds)
    w2 = max(2, W // ds)
    small = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)

    k = int(extra_gauss)
    if k % 2 == 0:
        k += 1
    k = min(k, (min(H, W) // 2) * 2 - 1)
    if k >= 3:
        up = cv2.GaussianBlur(up, (k, k), 0)
    return up

def resize_rgba_like(rgb, a, new_w, new_h):
    rgb_r = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    a_r = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    a_r = np.clip(a_r, 0.0, 1.0).astype(np.float32)
    return rgb_r, a_r

def alpha_composite(bg, fg_rgb, fg_a, x, y):
    """
    composite fg_rgb on bg at (x,y) (top-left corner).
    """
    out = bg.copy()
    H, W = bg.shape[:2]
    h, w = fg_a.shape[:2]

    x0, y0 = x, y
    x1, y1 = x + w, y + h
    tx0, ty0 = max(0, x0), max(0, y0)
    tx1, ty1 = min(W, x1), min(H, y1)
    if tx0 >= tx1 or ty0 >= ty1:
        return out
    fx0, fy0 = tx0 - x0, ty0 - y0
    fx1, fy1 = fx0 + (tx1 - tx0), fy0 + (ty1 - ty0)

    patch_bg = out[ty0:ty1, tx0:tx1].astype(np.float32)
    patch_fg = fg_rgb[fy0:fy1, fx0:fx1].astype(np.float32)
    patch_a = fg_a[fy0:fy1, fx0:fx1][..., None].astype(np.float32)

    patch = patch_bg * (1.0 - patch_a) + patch_fg * patch_a
    out[ty0:ty1, tx0:tx1] = np.clip(patch, 0, 255).astype(np.uint8)
    return out

def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

def make_smooth_lerp_video(img0, img1, num_frames = 50):
    assert img0.dtype == np.uint8 and img1.dtype == np.uint8
    H, W, _ = img1.shape
    img0_r = cv2.resize(img0, (W, H), interpolation=cv2.INTER_CUBIC)

    frames = []
    img0_f, img1_f = img0_r.astype(np.float32), img1.astype(np.float32)

    for i in tqdm.tqdm(range(num_frames)):
        t = i / (num_frames - 1)
        w = smoothstep(t)

        frame = img0_f * (1.0 - w) + img1_f * w
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        frames.append(frame)

    return frames
def align_and_make_video(
    img,
    person,
    alpha,                       # background-weight alpha (0..1), edge-soft
    foot_y,
    bbox,
    frames,
    bg_blur_max,

    mystery_bg, 
                   # background WITH stand-in person (same size as bg_without_person)
    mystery_bg_without_person,   # clean background (target canvas)
    foot_y_m,
    bbox_m,

    out_video_path,
    num_frames = 120,
    fps = 30,

    # blur control
    blur_downsample = 16,
    blur_extra_gauss = 51,

    # crop padding
    crop_pad = 30,
):
    """
    return target_img (last frame), and write video.

    points:
    - final: foot_y align to foot_y_m
    - scale: scale by bbox height to match bbox_m height (keep aspect ratio)
    - x: align by bbox center x (more stable, and not depend on foot_x)
    - animation: background blur->sharp; person smooth move+scale to final position
    """

    bg_clean = mystery_bg_without_person #target canvas
    Ht, Wt = bg_clean.shape[:2]

    fg_alpha_full = (1.0 - alpha).astype(np.float32) #person weight
    fg_alpha_full = np.clip(fg_alpha_full, 0.0, 1.0)

    x0, y0, x1, y1 = bbox
    person_crop, (cx0, cy0, cx1, cy1) = crop_with_pad(person, x0, y0, x1, y1, pad=crop_pad)
    alpha_crop, _ = crop_with_pad(fg_alpha_full, x0, y0, x1, y1, pad=crop_pad)  # fg alpha crop

    foot_in_crop_y = float(foot_y - cy0)

    src_cx, _ = bbox_center(bbox)
    center_in_crop_x = float(src_cx - cx0)

    tgt_cx, _ = bbox_center(bbox_m)
    tgt_foot_y = float(foot_y_m)

    _, src_bbox_h = bbox_size(bbox)
    _, tgt_bbox_h = bbox_size(bbox_m)
    scale_end = tgt_bbox_h / max(1.0, src_bbox_h) * 1.2

    # compute start transform by mapping source image coordinates into target canvas via size ratio
    Hs, Ws = img.shape[:2]
    sx = Wt / float(Ws)
    sy = Ht / float(Hs)

    scale_start = sy

    center_start_x, foot_start_y = float(src_cx) * sx, float(foot_y) * sy
    scale_start, scale_end = float(np.clip(scale_start, 0.05, 5.0)), float(np.clip(scale_end, 0.2, 5.0))

    bg_blur_max_1 = very_blurry(bg_clean, downsample=blur_downsample, extra_gauss=blur_extra_gauss)

    frames_resized = [
        cv2.resize(f, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
        for f in frames
    ]
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"VP80"), fps, (Wt, Ht))
    for frame in frames_resized:
        writer.write(frame)
    target_img = None

    for i in tqdm.tqdm(range(num_frames)):
        t = i / (num_frames - 1 + 1e-8)

        # bg: blur -> sharp
        w_bg = ease_in_quad(t)  # 0:blur, 1:sharp
        bg_f = (bg_blur_max_1.astype(np.float32) * (1.0 - w_bg) +
                bg_clean.astype(np.float32) * w_bg)
        frame = np.clip(bg_f, 0, 255).astype(np.uint8)

        # person: smooth move to target
        w_move = ease_in_out_smooth(t)

        # current scale
        s = scale_start * (1.0 - w_move) + scale_end * w_move

        # current expected center x / foot y (in target canvas)
        cur_center_x = center_start_x * (1.0 - w_move) + float(tgt_cx) * w_move
        cur_foot_y = foot_start_y * (1.0 - w_move) + tgt_foot_y * w_move

        # resize crop by current scale
        ch, cw = alpha_crop.shape[:2]
        new_w = max(2, int(round(cw * s)))
        new_h = max(2, int(round(ch * s)))

        fg_rgb_s, fg_a_s = resize_rgba_like(person_crop, alpha_crop, new_w, new_h)

        # compute top-left corner of the crop in the target canvas
        x_tl = int(round(cur_center_x - center_in_crop_x * s))
        y_tl = int(round(cur_foot_y - foot_in_crop_y * s))

        frame = alpha_composite(frame, fg_rgb_s, fg_a_s, x_tl, y_tl)

        if i == 0:
            frames_1 = make_smooth_lerp_video(bg_blur_max, frame, num_frames=num_frames)
            for fframe in frames_1:
                writer.write(fframe)

        writer.write(frame)

        if i == num_frames - 1:
            target_img = frame

    # writer.release()
    assert target_img is not None
    model = Harmonizer(model_path="models/netG_epoch_80.pth")
    H, W = target_img.shape[:2]

    full_mask = np.zeros((H, W), dtype=np.float32)

    h, w = fg_a_s.shape[:2]
    x0, y0 = x_tl, y_tl
    x1, y1 = x0 + w, y0 + h

    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)

    fx0, fy0 = x0c - x0, y0c - y0
    fx1, fy1 = fx0 + (x1c - x0c), fy0 + (y1c - y0c)

    full_mask[y0c:y1c, x0c:x1c] = fg_a_s[fy0:fy1, fx0:fx1]

    target_img_pil = Image.fromarray(target_img).convert("RGB")
    mask_pil = Image.fromarray(
        (full_mask * 255).astype(np.uint8),
        mode="L"
    )
    print("Image Harmonization begin...")
    result = model.predict(target_img_pil, mask_pil)

    if hasattr(result, "detach"):
        result_np = result.detach().cpu()
        if result_np.dim() == 4:
            result_np = result_np[0]
        result_np = result_np.permute(1, 2, 0).numpy()
        result_np = (result_np * 255).clip(0, 255).astype(np.uint8)
    num_interp_frames = 50 #120
    for i in tqdm.tqdm(range(num_interp_frames)):
        t = i / (num_interp_frames - 1 + 1e-8)
        w = t

        interp = (
            (1.0 - w) * target_img.astype(np.float32) +
            w * result_np.astype(np.float32)
        )

        interp = np.clip(interp, 0, 255).astype(np.uint8)
        writer.write(interp)
    for _ in range(20):
        writer.write(result_np)

    smooth = cv2.bilateralFilter(result_np, 9, 75, 75)

    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)

    gray = cv2.medianBlur(gray, 3)

    edges = cv2.adaptiveThreshold(gray, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    # edges = edges.astype(np.uint8)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for i in tqdm.tqdm(range(60)):
        t = i / (60 - 1 + 1e-8)
        w = t
        interp = (
            (1.0 - w) * result_np.astype(np.float32) +
            w * edges.astype(np.float32)
        )
        interp = np.clip(interp, 0, 255).astype(np.uint8)
        writer.write(interp)

    for _ in tqdm.tqdm(range(60)):
        frame = edges.copy()
        writer.write(frame)

    writer.release()
    return result_np
def process_image(img, beast_id = 'lei'):
    H, W = img.shape[:2]
    mystery_bg = gen_background(gen = True, aspect = W/H, beast_id = beast_id)
    mystery_bg = cv2.imdecode(
        np.frombuffer(mystery_bg, np.uint8),
        cv2.IMREAD_COLOR
    )
    print("mystery_bg generated")
    
    mystery_bg_without_person = modify_image(mystery_bg, gen = True, aspect = W/H)
    mystery_bg_without_person = cv2.imdecode(
        np.frombuffer(mystery_bg_without_person, np.uint8),
        cv2.IMREAD_COLOR
    )
    print("mystery_bg_without_person generated")
    segmentor = SelfieSegmentor("models/selfie_multiclass_256x256.tflite")
    # monster, environment_mon, alpha_mon, foot_y_mon, bbox_mon = segment_person_with_edge_alpha(mystery_bg_without_person, segmentor)
    # return monster
    person, environment, alpha, foot_y, bbox = segment_person_with_edge_alpha(img, segmentor)
    # cv2.imwrite("alpha.jpg", alpha * 255)
    # return person
    frames, bg_blur_max = make_bg_blur_video(person, environment)
    person_m, environment_m, alpha_m, foot_y_m, bbox_m = segment_person_with_edge_alpha(mystery_bg, segmentor)
    vis_person = draw_bbox(img, bbox, color=(0, 0, 255), thickness=3)
    vis_person_m = draw_bbox(mystery_bg, bbox_m, color=(0, 0, 255), thickness=3)
    # cv2.imwrite("vis_person.jpg", vis_person)
    # cv2.imwrite("vis_person_m.jpg", vis_person_m)

    print(foot_y)
    target_img = align_and_make_video(
        img=img,
        person=person,
        alpha=alpha,
        foot_y=foot_y,
        bbox=bbox,
        frames=frames,
        bg_blur_max=bg_blur_max,
        mystery_bg=mystery_bg,
        mystery_bg_without_person=mystery_bg_without_person,
        foot_y_m=foot_y_m,
        bbox_m=bbox_m,
        out_video_path="output/out.webm",
        num_frames=60, # 120
        fps=30,
        blur_downsample=16,
        blur_extra_gauss=51,
        crop_pad=40
    )

    cv2.imwrite("target.png", target_img)
    img_out = img
    return img_out
