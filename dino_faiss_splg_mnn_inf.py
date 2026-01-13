import os
import time
import pickle
import numpy as np
from PIL import Image
import MNN
import faiss
import cv2
from typing import Union, List, Optional
import torchvision.transforms as T
from MNN import nn, expr
import gc
import time
from typing import Literal
import math

########################################################
# --- 加载模型 ---
########################################################
def load_dinov2_model(dino_model_path = "dinov2_vits14.mnn"):
    print("Loading DINOv2 MNN model...")
    dino_interpreter = MNN.Interpreter(dino_model_path)
    dino_session = dino_interpreter.createSession()
    dino_input = dino_interpreter.getSessionInput(dino_session)
    dino_model = {
        'interpreter': dino_interpreter,
        'session': dino_session, 
        'input': dino_input
    }
    return dino_model

def load_extractor_model(extractor_path = 'superpoint.mnn'):
    extractor = MNN.nn.load_module_from_file(
        extractor_path,
        ["image"],
        ["keypoints", "scores", "descriptors"]
    )
    return extractor

def load_lightglue_model(lightglue_path = 'superponit_lightglue.trt.mnn'):
    lightglue = MNN.nn.load_module_from_file(
        lightglue_path,
        ["kpts0", "kpts1", "desc0", "desc1"],
        ["matches0", "mscores0"]
    )
    return lightglue

########################################################
# --- 图像处理/数据处理 ---
########################################################
# --- 与 PyTorch 版相同的图像预处理 ---
def transform_dino_pil(img_path,):
    img = Image.open(img_path).convert("RGB")
    # Resize：保持比例最短边=256
    w, h = img.size
    if w < h:
        new_w, new_h = 224, int(h * 224 / w)
    else:
        new_h, new_w = 224, int(w * 224 / h)
    img = img.resize((new_w, new_h))
    
    # CenterCrop
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    
    # ToTensor + Normalize
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = (img - mean) / std
    return img


def normalize_keypoints(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
    size = np.array([w, h])
    shift = size / 2
    scale = size.max() / 2
    kpts = (kpts - shift) / scale
    return kpts.astype(np.float32)


def read_image(path: str, grayscale: bool = False) -> np.ndarray:
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(path, mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def resize_image(image: np.ndarray, size: Union[List[int], int], fn: str, interp: Optional[str] = "area"):
    h, w = image.shape[:2]
    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def normalize_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))
    elif image.ndim == 2:
        image = image[None]
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return image / 255.0


def load_image(path: str, grayscale=False, resize=None, fn="max", interp="area"):
    img = read_image(path, grayscale)
    scales = [1, 1]
    if resize is not None:
        img, scales = resize_image(img, resize, fn=fn, interp=interp)
    return normalize_image(img)[None].astype(np.float32), np.asarray(scales)

########################################################
# --- DINOv2 模型推理/fasis 索引 ---
########################################################
def extract_dino_feature_mnn(img_path, dino_interpreter, dino_session, dino_input):
    x = transform_dino_pil(img_path)
    x = np.expand_dims(x, axis=0).astype(np.float32)

    tmp_input = MNN.Tensor(x.shape, MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe)
    dino_input.copyFrom(tmp_input)
    dino_interpreter.runSession(dino_session)
    out_tensor = dino_interpreter.getSessionOutput(dino_session)
    feat = np.array(out_tensor.getData(), dtype=np.float32)
    feat = feat.reshape(-1)
    feat = feat / np.linalg.norm(feat, keepdims=True)
    return feat.astype(np.float32)
def build_dino_faiss(image_dir, dino_model, save_prefix="dino" , dino_save_features = None):

    features, paths = [], []

    if isinstance(dino_save_features, str) == False:
        fnames = list_images_by_depth(image_dir)
        for idx, fname in enumerate(fnames):
            if idx % 100 == 0:
                print(f"[DINO-MNN] Processing {idx}/{len(fnames)}...")
            path = os.path.join(image_dir, fname).replace("\\", "/")
            feat = extract_dino_feature_mnn(path, dino_model['interpreter'], dino_model['session'], dino_model['input'])
            features.append(feat)
            paths.append(path)
        if dino_save_features == True:
            with open(f"{save_prefix}_features.pkl", "wb") as f:
                pickle.dump(features, f)
            # with open(f"{save_prefix}_paths.pkl", "wb") as f:
            #     pickle.dump(paths, f)

    if isinstance(dino_save_features, str):
        with open(dino_save_features, "rb") as f:
            features = pickle.load(f)
        with open(f"{save_prefix}_paths.pkl", "rb") as f:
            paths = pickle.load(f)

    if features == []:
        raise ValueError("No features found.")
        
    features = np.stack(features).astype('float32')
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    dim = features.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(features)
    print(f"[DINO-MNN] Indexed {len(paths)} images.")
    faiss.write_index(index, f"{save_prefix}_index.faiss")
    with open(f"{save_prefix}_paths.pkl", "wb") as f:
        pickle.dump(paths, f)

def build_dino_faiss_ivfsq8(
    image_dir,
    dino_model,
    save_prefix="dino",
    nlist=64,               # IVF 簇数
    nprobe=8,               # 查询时探测多少个簇
    dino_save_features=None
):
    features, paths = [], []

    # -------------------------
    # 1) 加载或提取特征
    # -------------------------
    if isinstance(dino_save_features, str) == False:
        fnames = list_images_by_depth(image_dir)
        for idx, fname in enumerate(fnames):
            if idx % 100 == 0:
                print(f"[DINO-MNN] Processing {idx}/{len(fnames)}...")
            path = os.path.join(image_dir, fname).replace("\\", "/")
            feat = extract_dino_feature_mnn(
                path,
                dino_model['interpreter'],
                dino_model['session'],
                dino_model['input']
            )
            features.append(feat)
            paths.append(path)

        if dino_save_features == True:
            with open(f"{save_prefix}_features.pkl", "wb") as f:
                pickle.dump(features, f)

    if isinstance(dino_save_features, str):
        with open(dino_save_features, "rb") as f:
            features = pickle.load(f)
        with open(f"{save_prefix}_paths.pkl", "rb") as f:
            paths = pickle.load(f)

    if not features:
        raise ValueError("No features found.")

    # -------------------------
    # 2) 特征归一化
    # -------------------------
    features = np.stack(features).astype("float32")
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    dim = features.shape[1]

    print(f"[DINO-IVFSQ8] Building IVFSQ8 index: dim={dim}, nlist={nlist}, nprobe={nprobe}")

    # -------------------------
    # 3) 构造 SQ8 (Scalar Quantizer)
    #    SQ8 量化器使用 L2 metric
    # -------------------------
    quantizer = faiss.IndexFlatIP(dim)    # coarse quantizer 仍然用 IP

    # SQ8 量化：QT_8bit → 每维 8 bit
    index = faiss.IndexIVFScalarQuantizer(
        quantizer,
        dim,
        nlist,
        faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_INNER_PRODUCT
    )

    # IVF 查询参数
    index.nprobe = nprobe

    # -------------------------
    # 4) 训练
    # -------------------------
    print("[DINO-IVFSQ8] Training IVF-SQ8...")
    index.train(features)

    # -------------------------
    # 5) 添加特征
    # -------------------------
    print("[DINO-IVFSQ8] Adding features...")
    index.add(features)

    print(f"[DINO-IVFSQ8] Indexed {len(paths)} images.")

    # -------------------------
    # 6) 保存索引
    # -------------------------
    out_file = f"{save_prefix}_ivfsq8_index.faiss"
    faiss.write_index(index, out_file)

    with open(f"{save_prefix}_paths.pkl", "wb") as f:
        pickle.dump(paths, f)

    print(f"[DINO-IVFSQ8] Index saved to {out_file}")

def build_dino_faiss_ivf(
    image_dir,
    dino_model,
    save_prefix="dino",
    nlist=64,              # IVF coarse clusters, 6000条数据用64最适合
    nprobe=8,              # IVF 搜索时探测多少个簇
    dino_save_features=None
):
    features, paths = [], []

    # -------------------------
    # 1) 加载或提取特征
    # -------------------------
    if isinstance(dino_save_features, str) == False:
        fnames = list_images_by_depth(image_dir)
        for idx, fname in enumerate(fnames):
            if idx % 100 == 0:
                print(f"[DINO-MNN] Processing {idx}/{len(fnames)}...")
            path = os.path.join(image_dir, fname).replace("\\", "/")
            feat = extract_dino_feature_mnn(
                path,
                dino_model['interpreter'],
                dino_model['session'],
                dino_model['input']
            )
            features.append(feat)
            paths.append(path)

        if dino_save_features == True:
            with open(f"{save_prefix}_features.pkl", "wb") as f:
                pickle.dump(features, f)

    if isinstance(dino_save_features, str):
        with open(dino_save_features, "rb") as f:
            features = pickle.load(f)
        with open(f"{save_prefix}_paths.pkl", "rb") as f:
            paths = pickle.load(f)

    if not features:
        raise ValueError("No features found.")

    # -------------------------
    # 2) 特征归一化，用于余弦/IP
    # -------------------------
    features = np.stack(features).astype("float32")
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    dim = features.shape[1]

    print(f"[DINO-IVF] Building IVF index: dim={dim}, nlist={nlist}, nprobe={nprobe}")

    # -------------------------
    # 3) 构造 IVF(IP) 索引
    # -------------------------
    quantizer = faiss.IndexFlatIP(dim)  # IP quantizer（必须）
    
    index = faiss.IndexIVFFlat(
        quantizer,
        dim,
        nlist,
        faiss.METRIC_INNER_PRODUCT    # metric=IP
    )

    # print("metric (Python may show 0, ignore this bug):", index.metric_type)

    # IVF 参数
    index.nprobe = nprobe

    # -------------------------
    # 4) 训练 IVF（必须）
    # -------------------------
    print("[DINO-IVF] Training IVF clusters...")
    index.train(features)

    # -------------------------
    # 5) 添加特征
    # -------------------------
    print("[DINO-IVF] Adding features...")
    index.add(features)

    print(f"[DINO-IVF] Indexed {len(paths)} images.")

    # -------------------------
    # 6) 保存索引
    # -------------------------
    faiss.write_index(index, f"{save_prefix}_ivf_index.faiss")

    with open(f"{save_prefix}_paths.pkl", "wb") as f:
        pickle.dump(paths, f)

    print("[DINO-IVF] Index saved.")
    
def build_dino_faiss_ivfpq(
    image_dir,
    dino_model,
    save_prefix="dino",
    nlist=64,              # 6000 数据：64 最佳
    nprobe=8,              # IVF 搜索簇数
    M=32,                  # PQ分块数（256维时32最优）
    nbits=8,               # 每块编码位数（8bit 是最佳）
    dino_save_features=None
):
    features, paths = [], []

    # -------------------------
    # 1) 加载或提取特征
    # -------------------------
    if isinstance(dino_save_features, str) == False:
        fnames = list_images_by_depth(image_dir)
        for idx, fname in enumerate(fnames):
            if idx % 100 == 0:
                print(f"[DINO-MNN] Processing {idx}/{len(fnames)}...")
            path = os.path.join(image_dir, fname).replace("\\", "/")
            feat = extract_dino_feature_mnn(
                path,
                dino_model['interpreter'],
                dino_model['session'],
                dino_model['input']
            )
            features.append(feat)
            paths.append(path)

        if dino_save_features == True:
            with open(f"{save_prefix}_features.pkl", "wb") as f:
                pickle.dump(features, f)

    if isinstance(dino_save_features, str):
        with open(dino_save_features, "rb") as f:
            features = pickle.load(f)
        with open(f"{save_prefix}_paths.pkl", "rb") as f:
            paths = pickle.load(f)

    if not features:
        raise ValueError("No features found.")

    # -------------------------
    # 2) 特征归一化（余弦 / IP）
    # -------------------------
    features = np.stack(features).astype("float32")
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    dim = features.shape[1]

    print(f"[DINO-IVFPQ] Building IVFPQ index: dim={dim}, M={M}, nbits={nbits}, "
          f"nlist={nlist}, nprobe={nprobe}")

    # -------------------------
    # 3) 构造 IVFPQ（PQ32×8bit）
    # -------------------------
    quantizer = faiss.IndexFlatIP(dim)  # coarse quantizer（IP）

    index = faiss.IndexIVFPQ(
        quantizer,
        dim,
        nlist,
        M,          # PQ 分块数：256维→32块，每块8维
        nbits,      # 每块8bit
        faiss.METRIC_INNER_PRODUCT
    )

    # IVF 参数：探测簇个数
    index.nprobe = nprobe

    # -------------------------
    # 4) 训练 IVF + PQ（必须）
    # -------------------------
    print("[DINO-IVFPQ] Training IVF-PQ...")
    index.train(features)

    # -------------------------
    # 5) 添加全部向量
    # -------------------------
    print("[DINO-IVFPQ] Adding features...")
    index.add(features)

    print(f"[DINO-IVFPQ] Indexed {len(paths)} images.")

    # -------------------------
    # 6) 保存索引 + 路径
    # -------------------------
    faiss.write_index(index, f"{save_prefix}_ivfpq_index.faiss")

    with open(f"{save_prefix}_paths.pkl", "wb") as f:
        pickle.dump(paths, f)

    print("[DINO-IVFPQ] Index saved.")

def load_dino_index(prefix="dino", faiss_type: Literal[None, "ivf", "ivfpq", 'ivfsq8'] = None):
    if faiss_type == "ivfpq":
        index = faiss.read_index(f"{prefix}_ivfpq_index.faiss")
    elif faiss_type == "ivf":
        index = faiss.read_index(f"{prefix}_ivf_index.faiss")
    elif faiss_type == 'ivfsq8':
        index = faiss.read_index(f"{prefix}_ivfsq8_index.faiss")
    else:
        index = faiss.read_index(f"{prefix}_index.faiss")
    # index = faiss.read_index(f"{prefix}_index.faiss")
    with open(f"{prefix}_paths.pkl", "rb") as f:
        paths = pickle.load(f)
    print(f"[DINO-MNN] Loaded index with {index.ntotal} vectors.")
    return index, paths

def dino_search(query_path, index, paths, dino_model, topk=10):
    start = time.time()
    q_feat = extract_dino_feature_mnn(query_path, dino_model['interpreter'], dino_model['session'], dino_model['input'])[None, :]
    q_feat = q_feat / np.linalg.norm(q_feat, axis=1, keepdims=True)
    scores, ids = index.search(q_feat, topk)
    end = time.time()
    print(f"[DINO-MNN] Query processed in {end - start:.3f}s")
    results = []
    for i, idx in enumerate(ids[0]):
        results.append({'path': paths[idx], 'score': scores[0][i]})
        print(f"Top {i+1}: {paths[idx]} (score={scores[0][i]:.3f})")
    return results

########################################################
# --- SuperPoint-LihghtGlue 模型推理 ---
########################################################

def process_superpoint_output(kpts, scores, desc, score_thresh=0.2, topk=2048):
    """
    处理 SuperPoint 输出的关键点、得分和描述子
    支持输入为 (N, 2)/(1, N, 2) 两种形式
    """

    l = kpts.shape[1]
    # 如果带 batch 维，则去掉 batch 维
    if kpts.ndim == 3 and kpts.shape[0] == 1:
        kpts = kpts[0]
        scores = scores[0]
        desc = desc[0]

    assert kpts.shape[0] == scores.shape[0] == desc.shape[0], \
        f"Shape mismatch: {kpts.shape}, {scores.shape}, {desc.shape}"

    # Step 1: 根据分数阈值过滤
    mask = scores > score_thresh
    kpts, scores, desc = kpts[mask], scores[mask], desc[mask]

    if len(scores) == 0:
        print(f"⚠️ 没有找到 score>{score_thresh} 的关键点")
        D = desc.shape[1] if desc.ndim == 2 else 256
        return np.empty((1, 0, 2)), np.empty((1, 0)), np.empty((1, 0, D))

    # Step 2: 按分数排序
    order = np.argsort(scores)[::-1]
    if len(order) > topk:
        order = order[:topk]

    kpts_f = kpts[order]
    scores_f = scores[order]
    desc_f = desc[order]

    # 重新加上 batch 维
    kpts_f = kpts_f[None, ...].astype(np.float32)     # (1, M, 2)
    scores_f = scores_f[None, ...].astype(np.float32) # (1, M)
    desc_f = desc_f[None, ...].astype(np.float32)     # (1, M, D)

    # print(f"✅ 保留 {kpts_f.shape[1]} 个关键点,删除{l - kpts_f.shape[1]}个关键点,(score>{score_thresh}, topk={topk})")
    return kpts_f, scores_f, desc_f

def quantize_int8_per_token(x):
    """
    支持 (T, C) 或 (B, T, C)
    """
    if x.ndim == 2:
        x = x[None, ...]  # 变成 (1, T, C)

    B, T, C = x.shape
    scale = np.max(np.abs(x), axis=2) / 127.0  # (B, T)
    scale = scale.astype(np.float16)
    scale[scale == 0] = 1e-8
    scale_bc = scale[..., None]                # (B, T, 1)
    x_int8 = np.round(x / scale_bc).astype(np.int8)
    
    return x_int8, scale

def quantize_int8_global(x):
    """
    x: float32 array of any shape
    return: int8 array, and global scale
    """
    max_abs = np.max(np.abs(x))
    scale = max_abs / 127.0
    scale = scale.astype(np.float16)
    if scale < 1e-8:
        scale = 1e-8  # 防止除零

    x_int8 = np.round(x / scale).astype(np.int8)
    return x_int8, scale

def quantize_uint8_global(x):
    """
    x: float32 array, values >=0  (e.g., range 0~1 or any 0~max)
    return:
        - uint8 array (same shape)
        - scale (float32)
    """
    max_val = np.max(x)
    if max_val < 1e-8:
        max_val = 1e-8  # avoid divide-by-zero

    scale = max_val / 255.0

    # 量化: 0~max → 0~255
    x_uint8 = np.round(x / scale).astype(np.uint8)

    return x_uint8, scale

def dequantize_uint8_global(x_uint8, scale):
    """
    x_uint8: uint8 array
    scale: float32
    return: float32 recovered array
    """
    return (x_uint8.astype(np.float32) * scale)

def dequantize_int8_global(x_int8, scale):
    return x_int8.astype(np.float32) * scale


def dequantize_int8(x_int8, scale):
    if scale.ndim == 1:  # (T,)
        scale = scale.reshape(1, -1, 1)
    elif scale.ndim == 2:  # (B, T)
        scale = scale[..., None]
    return x_int8.astype(np.float32) * scale


def extract_superpoint_feature_mnn(img_path, extractor, resize_size=512, type_int8=True):

    image, _ = load_image(img_path, resize = resize_size, grayscale=True)

    x = expr.const(image, image.shape, expr.NCHW, expr.float)

    feats = extractor.forward([x])

    kpts, scores, desc = [f.read() for f in feats]
    kpts, scores, desc = process_superpoint_output(kpts, scores, desc, score_thresh=0.1, topk=256)
    # type_int8 = True
    if type_int8:
        kpts_uint8, kpts_scale = quantize_uint8_global(kpts)
        # kpts_d = dequantize_uint8_global(kpts_uint8, kpts_scale)
        desc_int8, desc_scale = quantize_int8_global(desc)
        # desc_d = dequantize_int8_global(desc_int8, desc_scale)
        return {"keypoints": kpts_uint8, 'keypoints_scale': kpts_scale, "descriptors": desc_int8, "descriptors_scale": desc_scale, 'shape': image.shape}
        
    # kpts = kpts.astype(np.float16).copy()
    # desc = desc.astype(np.float16).copy()

    # x_kpts = expr.const(normalize_keypoints(kpts.astype(np.float32), image.shape[2], image.shape[3]), kpts.shape, expr.NCHW, expr.float)
    # x_desc = expr.const(desc.astype(np.float32), desc.shape, expr.NCHW, expr.float)
    # gc.collect()
    # print(type(kpts), kpts.flags['OWNDATA'])
    return {"keypoints": kpts, "descriptors": desc, 'shape': image.shape}

    # return {"keypoints": kpts}
    # return None

def build_superpoint_features(image_dir, extractor, resize_size=512, save_prefix="glue"):
    feats_dict, paths = {}, []
    fnames = list_images_by_depth(image_dir)
    for idx, fname in enumerate(fnames):
        if idx % 100 == 0:
            print(f"[SP-MNN] Processing {idx}/{len(fnames)}...")
        path = os.path.join(image_dir, fname).replace("\\", "/")
        # st = time.time()
        feats = extract_superpoint_feature_mnn(path, extractor, resize_size=resize_size)
        # print(f"[SP-MNN] {path} done in {time.time() - st:.2f}s")
        if feats is not None:
            feats_dict[path] = feats
            paths.append(path)

    # with open(f"{save_prefix}_features.pkl", "wb") as f:
    #     pickle.dump(feats_dict, f)

    with open(f"{save_prefix}_paths.pkl", "wb") as f:
        pickle.dump(paths, f)

        # 按paths顺序分组保存features
    total = len(paths)
    chunk_size = 1000
    num_chunks = math.ceil(total / chunk_size)
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        part_dict = {p: feats_dict[p] for p in paths[start:end]}
        part_name = f"{save_prefix}_features_{i:03d}.pkl"
        with open(part_name, "wb") as f:
            pickle.dump(part_dict, f)
        print(f"[SP-MNN] ✅ {part_name} ({start}~{end-1}) 共 {len(part_dict)} 条")

    print(f"[SP-MNN] Done, {len(paths)} features extracted.")

import multiprocessing, pickle, os

def save_worker(save_path, feats_dict):
    with open(save_path, "wb") as f:
        pickle.dump(feats_dict, f)
    print("✅ 子进程保存完成:", save_path)

def convert_feature_to_ver(feature, type_int8=True):

    if type_int8:
        feature['keypoints'] = dequantize_uint8_global(feature['keypoints'], feature['keypoints_scale'])
        feature['descriptors'] = dequantize_int8_global(feature['descriptors'], feature['descriptors_scale'])
    feature['keypoints'] = expr.const(normalize_keypoints(feature['keypoints'].astype(np.float32), feature['shape'][2], feature['shape'][3]), feature['keypoints'].shape, expr.NCHW, expr.float).copy()
    feature['descriptors'] = expr.const(feature['descriptors'].astype(np.float32), feature['descriptors'].shape, expr.NCHW, expr.float).copy()
    return feature

def load_glue_features(prefix="glue", index = 0):
    # with open(f"{prefix}_paths.pkl", "rb") as f:
    #     paths = pickle.load(f)
    with open(f"{prefix}_features_{index:03d}.pkl", "rb") as f:
        features = pickle.load(f)
    print(f"[SP-MNN] Loaded {len(features)} feature maps.")

    for feature in features.values():
        feature = convert_feature_to_ver(feature)
    return features

def load_glue_index(prefix="glue"):
    with open(f"{prefix}_paths.pkl", "rb") as f:
        paths = pickle.load(f)
    # with open(f"{prefix}_features.pkl", "rb") as f:
    #     features = pickle.load(f)
    # print(f"[SP-MNN] Loaded {len(features)} feature maps.")

    # for feature in features.values():
    #     feature = convert_feature_to_ver(feature)
    return paths# , features

def glue_match_features(feats0, feats1, lightglue, min_matches=600, score_threshold=0.7):
    # 转换为 torch tensor 形式供 LightGlue 使用
    # x_kpts0 = feats0['keypoints']
    # x_kpts1 = feats1['keypoints']
    # x_desc0 = feats0['descriptors']
    # x_desc1 = feats1['descriptors']

    inputs = [feats0['keypoints'],feats1['keypoints'],feats0['descriptors'],feats1['descriptors']]
    
    outputs = lightglue.forward(inputs)
    if len(outputs) != 2: # 兼容 LightGlue 旧版本
        return False, -1, -1

    matches0, mscores0 = outputs
    
    matches = np.array(matches0.read(), dtype=np.int32)
    mscores = np.array(mscores0.read(), dtype=np.float32).squeeze()

    # 过滤非法匹配（若存在 -1）
    valid = (matches[:, 0] >= 0) & (matches[:, 1] >= 0)
    if matches.size > 1:
        matches = matches[valid]
    else:
        matches = np.array([matches])
    if mscores.size > 1:
        mscores = mscores[valid]
    else:
        mscores = np.array([mscores])

    n_matches = len(mscores)
    mean_score = mscores.mean()
    
    same = False
    # if n_matches > min_matches and mean_score > score_threshold: # 最低匹配基准
    #     same = True
    if n_matches > 0.65 * min(feats0['keypoints'].shape[1], feats1['keypoints'].shape[1]) and mean_score > 0.7: # 高基准匹配
        same = True
    if n_matches > 0.8 * min(feats0['keypoints'].shape[1], feats1['keypoints'].shape[1]) and mean_score > 0.6: # 高数量匹配
        same = True
    if mean_score > 0.9 and n_matches > 0.2 * min(feats0['keypoints'].shape[1], feats1['keypoints'].shape[1]) and n_matches > 10: # 高置信度匹配
        same = True
    if mean_score * n_matches > score_threshold * min_matches: # 置信度*数量匹配基准
        same = True
    return same, n_matches, mean_score

########################################################
def list_images_by_depth(base_dir, depth=0, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")):
    """
    列出指定目录下的图像文件路径（返回相对路径）。

    参数：
        base_dir (str): 起始目录（相对路径或绝对路径）
        depth (int): 要列出的目录层级
                     0 = 递归列出所有子目录
                     1 = 只列出当前目录
                     2 = 列出当前目录及第一级子文件夹中的图像 ...
        exts (tuple): 支持的图像扩展名（默认常见几种）

    返回：
        list[str]: 从 base_dir 开始的相对路径列表（文件路径）
    """
    if not os.path.exists(base_dir):
        return []

    base_dir = os.path.abspath(base_dir)
    result = []
    base_depth = base_dir.rstrip(os.sep).count(os.sep)

    for root, dirs, files in os.walk(base_dir):
        current_depth = root.rstrip(os.sep).count(os.sep) - base_depth

        # 如果指定了层级限制且超过则跳过
        if depth != 0 and current_depth >= depth:
            dirs[:] = []  # 不再深入该子目录
            continue

        for f in files:
            if f.lower().endswith(exts):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, base_dir)
                result.append(rel_path.replace("\\", "/"))  # 转为统一分隔符

    return result

def generate_faiss_index_and_sp_features(
        image_folder, 
        data_name, 
        data_dir = None,
        sp_size = 512, 
        dino_model_path = "dinov2_vits14.mnn",
        extractor_model_path = "superpoint.mnn",
        dino_save_features = None,
        faiss_type: Literal[None, "ivf", "ivfpq", 'ivfsq8'] = None,
        **kwargs
        # lightglue_model_path = "superpoint_lightglue.trt.mnn"
    ):

    dino = load_dinov2_model(dino_model_path)
    extractor = load_extractor_model(extractor_model_path)
    # lightglue = load_lightglue_model(lightglue_model_path)

    dino_prefix = f'dino_mnn_{data_name}' if data_name else 'dino_mnn'
    glue_prefix = f'glue_mnn_{sp_size}_{data_name}' if data_name else f'glue_mnn_{sp_size}'
    if data_dir:
        dino_prefix = f'{data_dir}/{dino_prefix}'
        glue_prefix = f'{data_dir}/{glue_prefix}'

    if faiss_type == None:
        build_dino_faiss(image_folder, dino, save_prefix=dino_prefix, dino_save_features = dino_save_features)
    elif faiss_type == "ivf":
        nlist = kwargs.get('nlist', 64)             
        nprobe = kwargs.get('nprobe', 16)
        build_dino_faiss_ivf(image_folder, dino, save_prefix=dino_prefix, nlist=nlist, nprobe=nprobe, dino_save_features = dino_save_features)
    elif faiss_type == "ivfpq":
        nlist = kwargs.get('nlist', 64)
        m = kwargs.get('M', 32)
        nprobe = kwargs.get('nprobe', 16)
        nbits = kwargs.get('nbits', 8) 
        build_dino_faiss_ivfpq(image_folder, dino, save_prefix=dino_prefix, nlist=nlist, nprobe=nprobe, M=m, nbits=nbits, dino_save_features = dino_save_features)
    elif faiss_type == "ivfsq8":
        nlist = kwargs.get('nlist', 64)
        nprobe = kwargs.get('nprobe', 16)
        build_dino_faiss_ivfsq8(image_folder, dino, save_prefix=dino_prefix, nlist=nlist, nprobe=nprobe, dino_save_features = dino_save_features)
    build_superpoint_features(image_folder, extractor, resize_size=sp_size, save_prefix=glue_prefix)


def search_image_sence(
        query_image, 
        data_name,
        dino,
        extractor,
        lightglue,
        dino_index,
        dino_paths,
        glue_paths,
        indxl,
        glue_features,
        sp_size=512,
        stop_mode: Literal["first", "end"] = "first",
        dino_topk = 10,
        dino_threshold = 0.7,
        relative_1024_min_matches = 400,
        relative_1024_score_threshold = 0.8,
    ):

    # dino = load_dinov2_model(dino_model_path)
    # extractor = load_extractor_model(extractor_model_path)
    # lightglue = load_lightglue_model(lightglue_model_path)

    # dino_prefix = f'dino_mnn_{data_name}' if data_name else 'dino_mnn'
    # glue_prefix = f'glue_mnn_{sp_size}_{data_name}' if data_name else f'glue_mnn_{sp_size}'

    # dino_index, dino_paths = load_dino_index(dino_prefix)
    # glue_paths, glue_features = load_glue_index(glue_prefix)

    start_time = time.time()
    results = dino_search(query_image, dino_index, dino_paths, dino, topk=dino_topk)
    results = [r for r in results if r['score'] > dino_threshold]

    # ========== 验证匹配 ========== #
    query_feats = extract_superpoint_feature_mnn(query_image, extractor, resize_size=sp_size)
    query_feats = convert_feature_to_ver(query_feats)
    true_count = 0
    match_name = None
    # indxl = glue_paths.index(results[0]['path'])
    # glue_features = load_glue_features(glue_prefix, indxl // 1000)
    for r in results:
        print(f"\n==== 验证 {r['path']} ====")
        indx = glue_paths.index(r['path']) // 1000
        if indx != indxl:
            glue_features = load_glue_features(glue_prefix, indx)
            indxl = indx
        same, n, s = glue_match_features(query_feats, glue_features[r['path']], lightglue, min_matches=relative_1024_min_matches*(sp_size/1024), score_threshold=max(relative_1024_score_threshold*(sp_size/1024), 0.33))
        print(f"{'✅ 同一场景' if same else '❌ 不同场景'} ({n} matches, mean={s:.3f})")
        if same:
            true_count += 1
            match_name = r['path']
            if stop_mode == 'first': break
    
    print(f"\n✅ Total matches: {true_count}/{len(results)}")
    end_time = time.time() 
    print(f"✅ 耗时: {end_time - start_time:.3f}s")
    if true_count > 0:
        print(f"✅ 验证通过")
        print(f'Match Name: {match_name}')
        return True, match_name, indxl, glue_features, end_time - start_time
    else:
        print(f"❌ 验证失败")
        return False, match_name, indxl, glue_features, end_time - start_time
    

import pyautogui
from PIL import Image
import os
import time

def capture_center_169_once(save_dir="captures") -> str:
    """截取屏幕中心16:9区域并保存为720p PNG，返回文件路径"""
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    img = pyautogui.screenshot()
    w, h = img.size
    ratio = 16 / 9
    if w / h > ratio:
        new_w = int(h * ratio)
        new_h = h
    else:
        new_w = w
        new_h = int(w / ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    img = img.crop((left, top, left + new_w, top + new_h))
    img = img.resize((1280, 720), Image.LANCZOS)
    path = os.path.join(save_dir, f"cap_{timestamp}.png")
    img.save(path, "PNG")
    return path


if __name__ == "__main__":

    image_folder = "hwkfg3_frames"

    data_dir = 'data'
    data_name = "hwkfg3_nms"
    sp_size = 512
    stop_mode = 'first'# "first" "end"
    dino_topk = 5
    dino_threshold = 0.7
    relative_1024_min_matches = 200
    relative_1024_score_threshold = 0.8
    faiss_type = 'ivfsq8' # None / hnsw / ivfhnsw

    build = False
    danmu = True

    dino_model_path = "model/dinov2_vits14.mnn"
    extractor_model_path = "model/superpoint.mnn"
    lightglue_model_path = "model/superpoint_lightglue.trt.mnn"

    dino_save_features_path = f".pkl"
    dino_save_features = True # 'dino_mnn_1e6_n_cn_features.pkl' # None / path to load / True to save to default path

    nlist = 32 # ivf
    m = 32 # 32 # hnsw
    efConstruction = 80 # 200 # hnsw
    efSearch = 32 # 64 # hnsw
    nprobe = 16 # 16 # ivf
    nbits = 4 # ivf

    if build:
        generate_faiss_index_and_sp_features(
            image_folder = image_folder,
            sp_size = sp_size,
            data_dir = data_dir,
            data_name = data_name,
            dino_model_path = dino_model_path,
            extractor_model_path = extractor_model_path,
            dino_save_features = dino_save_features,
            faiss_type = faiss_type,
            nlist = nlist,
            M = m,    
            nbits = nbits, 
            efConstruction = efConstruction,
            efSearch = efSearch, 
            nprobe = nprobe,
        )

    dino = load_dinov2_model(dino_model_path)
    extractor = load_extractor_model(extractor_model_path)
    lightglue = load_lightglue_model(lightglue_model_path)

    dino_prefix = f'dino_mnn_{data_name}' if data_name else 'dino_mnn'
    glue_prefix = f'glue_mnn_{sp_size}_{data_name}' if data_name else f'glue_mnn_{sp_size}'
    if data_dir:
        dino_prefix = f'{data_dir}/{dino_prefix}'
        glue_prefix = f'{data_dir}/{glue_prefix}'

    dino_index, dino_paths = load_dino_index(dino_prefix, faiss_type)
    glue_paths = load_glue_index(glue_prefix)
    indx_st = 0
    glue_features = load_glue_features(glue_prefix, indx_st)

    cnt = 0
    true = 0
    stime = 0

    
    while True:
        start_time = time.time()
        # p = list_images_by_depth(image_folder)
        # query_image = r"Hello World\2053.png"
        query_image = capture_center_169_once()


        res, match_name, indx_st, glue_features, ttime = search_image_sence(
            query_image = query_image,
            data_name = data_name,
            dino = dino,
            extractor = extractor,
            lightglue = lightglue,
            dino_index = dino_index,
            dino_paths = dino_paths,
            glue_paths = glue_paths,
            indxl=indx_st,
            glue_features = glue_features,
            # glue_features = glue_features,
            sp_size = sp_size,
            stop_mode = stop_mode,
            dino_topk = dino_topk,
            dino_threshold = dino_threshold,
            relative_1024_min_matches = relative_1024_min_matches,
            relative_1024_score_threshold = relative_1024_score_threshold
        )
        end_time = time.time()
        if res:
            true += 1
        cnt += 1
        stime += (end_time - start_time)
        print(f'平均时间：{stime/cnt:.2f}s')

        print(f'准确率：{true}/{cnt} = {true/cnt:.2%}')
        # time.sleep(1)


# if __name__ == "__main__":

#     image_folder = "mp4png"
#     # p = list_images_by_depth(image_folder)
#     query_image = r"mp4png\7\0007.png"

#     dino_model_path = "dinov2_vits14.mnn"
#     extractor_model_path = "superpoint.mnn"
#     lightglue_model_path = "superpoint_lightglue.trt.mnn"

#     sp_size = 512

#     data_name = "test01"
#     dino_prefix = f'dino_mnn_{data_name}' if data_name else 'dino_mnn'
#     glue_prefix = f'glue_mnn_{sp_size}_{data_name}' if data_name else f'glue_mnn_{sp_size}'

#     dino = load_dinov2_model(dino_model_path)
#     extractor = load_extractor_model(extractor_model_path)
#     lightglue = load_lightglue_model(lightglue_model_path)

#     # ========== 建索引 ========== #
    
#     build_dino_faiss(image_folder, dino, save_prefix=dino_prefix)
#     build_superpoint_features(image_folder, extractor, resize_size=sp_size, save_prefix=glue_prefix)

#     dino_index, dino_paths = load_dino_index(dino_prefix)
#     glue_paths, glue_features = load_glue_index(glue_prefix)

#     # ========== 检索 ========== #
#     start_time = time.time()
#     results = dino_search(query_image, dino_index, dino_paths, dino, topk=5)
#     results = [r for r in results if r['score'] > 0.8]

#     # ========== 验证匹配 ========== #
#     query_feats = extract_superpoint_feature_mnn(query_image, extractor, resize_size=sp_size)
#     query_feats = convert_feature_to_ver(query_feats)
#     true_count = 0
#     match_name = None
#     for r in results:
#         print(f"\n==== 验证 {r['path']} ====")
#         same, n, s = glue_match_features(query_feats, glue_features[r['path']], lightglue, min_matches=600*(sp_size/720), score_threshold=max(0.8*(sp_size/1024), 0.33))
#         print(f"{'✅ 同一场景' if same else '❌ 不同场景'} ({n} matches, mean={s:.3f})")
#         if same:
#             true_count += 1
#             match_name = r['path']
#             # break
    
#     print(f"\n✅ Total matches: {true_count}/{len(results)}")
#     if true_count > 0:
#         print(f"✅ 验证通过")
#         print(f'Match Name: {match_name}')
#     else:
#         print(f"❌ 验证失败")
#     print(f"✅ 耗时: {time.time() - start_time:.3f}s")
