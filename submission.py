from tqdm import tqdm
import random
import os
import zipfile
import json
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector
import open_clip
import warnings
import pandas as pd
import cv2
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

CFG = {
    'SUB_DIR' : './submission',
    'SEED' : 42
}
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['SEED']) 



# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load ControlNet for SDXL
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
).to(device)

# 2. Load SDXL base with ControlNet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)

# Optional: Use faster scheduler
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# 3. Define preprocessing (Canny)
def preprocess_canny(image: Image.Image, low_threshold=100, high_threshold=200) -> Image.Image:
    image_np = np.array(image)
    if image_np.ndim == 3 and image_np.shape[2] == 3:
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image_np
    edges = cv2.Canny(image_gray, low_threshold, high_threshold)
    edges_3ch = np.stack([edges]*3, axis=-1)  # Make 3-channel
    return Image.fromarray(edges_3ch)

# 4. Inference loop
test_df = pd.read_csv('./data/test.csv')
os.makedirs("./image_log/test_2025-07-31-16-50-14_ug_4.5", exist_ok=True)

out_imgs = []
out_img_names = []

out_imgs = []
out_img_names = []

for img_id in test_df['ID']:
    img_path = f'./image_log/test_output/{img_id}.png'

    if not os.path.exists(img_path):
        print(f"❌ 파일 없음: {img_path}")
        continue

    img = Image.open(img_path).convert("RGB")
    out_imgs.append(img)
    out_img_names.append(img_id)

# 추론 결과물 디렉토리 생성
os.makedirs(CFG['SUB_DIR'], exist_ok=True)
# **중요** 추론 이미지 평가용 Embedding 추출 모델
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai") # 모델명을 반드시 일치시켜야합니다.
clip_model.to(device)
# 평가 제출을 위해 추론된 이미지들을 ViT-L-14 모델로 임베딩 벡터(Feature)를 추출합니다.
feat_imgs = []
for output_img, img_id in tqdm(zip(out_imgs, out_img_names)):
    path_out_img = CFG['SUB_DIR'] + '/' + img_id + '.png' 
    output_img.save(path_out_img)
    # 평가용 임베딩 생성 및 저장
    output_img = clip_preprocess(output_img).unsqueeze(0).cuda()
    with torch.no_grad():
        feat_img = clip_model.encode_image(output_img)
        feat_img /= feat_img.norm(dim=-1, keepdim=True) # L2 정규화 필수

    feat_img = feat_img.detach().cpu().numpy().reshape(-1)
    feat_imgs.append(feat_img)
feat_imgs = np.array(feat_imgs)
vec_columns = [f'vec_{i}' for i in range(feat_imgs.shape[1])]
feat_submission = pd.DataFrame(feat_imgs, columns=vec_columns)
feat_submission.insert(0, 'ID', out_img_names)
feat_submission.to_csv(CFG['SUB_DIR']+'/embed_submission.csv', index=False)


# 최종 제출물 (ZIP) 생성 경로
# 제출물 (ZIP) 내에는 디렉토리(폴더)가 없이 구성해야합니다.
zip_path = './submissions.zip'

# zip 파일 생성
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_name in os.listdir(CFG['SUB_DIR']):
        file_path = os.path.join(CFG['SUB_DIR'], file_name)

        # 일반 파일이며 숨김 파일이 아닌 경우만 포함
        if os.path.isfile(file_path) and not file_name.startswith('.'):
            zipf.write(file_path, arcname=file_name)

print(f"✅ 압축 완료: {zip_path}")
