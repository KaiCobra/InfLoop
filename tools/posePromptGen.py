import itertools
import json
import os

# ==========================================
# 1. 定義三大維度的參數字典
# ==========================================
# body_structure = 'head and shoulders portrait'
camera = 'head and shoulders portrait, photography'
subject_id = [
    "boy",
    "girl",
    "woman",
    "man"
]

# views = [
#     "frontal view",
#     "three-quarter view",
#     "side view",
#     "back view"
# ]
# race = [
#     "Caucasian",
#     "African",
#     "Asian",
#     "Hispanic",
#     "Middle Eastern",
# ]

# age = [
#     'young',
#     'eldly'
# ]

poses = [
    # "head facing forward straight",
    # "head turned to his/her left",
    # "head turned to his/her right",
    # "head turned to his/her left over the shoulder",
    # "head turned to his/her right over the shoulder",
    # "head turned to his/her left and tilted up",
    # "head turned to his/her left and tilted down",
    # "head turned to his/her right and tilted up",
    # "head turned to his/her right and tilted down",
    # "head tilted up",
    # "head tilted down",
    # "head turned to his/her left over the shoulder and tilted up",
    # "head turned to his/her left over the shoulder and tilted down",
    # "head turned to his/her right over the shoulder and tilted up",
    # "head turned to his/her right over the shoulder and tilted down",

    "turned his/her head to his/her left and tilted up",
    "turned his/her head to his/her left and tilted down",
    "turned his/her head to his/her left and looking straight",
    "turned his/her head to his/her right and tilted up",
    "turned his/her head to his/her right and tilted down",
    "turned his/her head to his/her right and looking straight",
    "turned his/her head to his/her left over the shoulder and tilted up",
    "turned his/her head to his/her left over the shoulder and tilted down",
    "turned his/her head to his/her left over the shoulder and looking straight",
    "turned his/her head to his/her right over the shoulder and tilted up",
    "turned his/her head to his/her right over the shoulder and tilted down",
    "turned his/her head to his/her right over the shoulder and looking straight",
    "facing forward and tilted up",
    "facing forward and tilted down",
    "facing forward and looking straight",
    
    # "looking fully backward over the left shoulder, extreme head turn", 
    # "looking fully backward over the left shoulder, extreme head turn, chin-raised high", 
    # "looking fully backward over the left shoulder, extreme head turn, chin-tucked deep",
    # "looking fully backward over the right shoulder, extreme head turn",
    # "looking fully backward over the right shoulder, extreme head turn, chin-raised high", 
    # "looking fully backward over the right shoulder, extreme head turn, chin-tucked deep"
]
    # "head turned over the left shoulder",
    # "head turned over the right shoulder",    
    # "head turned over the left shoulder and tilted up",
    # "head turned over the left shoulder and tilted down",
    # "head turned over the right shoulder and tilted up",
    # "head turned over the right shoulder and tilted down"

    # "backward-glancing over the left shoulder", 
    # "backward-glancing over the left shoulder, chin-raised", 
    # "backward-glancing over the left shoulder, chin-tucked",
    # "backward-glancing over the right shoulder",
    # "backward-glancing over the right shoulder, chin-raised", 
    # "backward-glancing over the right shoulder, chin-tucked",



# gazes = [
#     "look directly",
#     "looking up",
#     "looking down",
#     "looking left",
#     "looking right",
#     "looking top-left",
#     "looking top-right",
#     "looking bottom-left",
#     "looking bottom-right"
# ]

# ==========================================
# 2. 進行排列組合與 Prompt 生成
# ==========================================
generated_data = []

# 使用 itertools.product 自動產生所有可能的組合
for subject_id, pose in itertools.product(subject_id, poses):
    # 套用我們設定的 Prompt 框架
    prompt_text = f"{camera}, a {subject_id} {pose}."
    print(f"Cureent id: {subject_id}")
    if subject_id == 'boy' or subject_id == 'man':
        prompt_text = prompt_text.replace("his/her", "his")
    elif subject_id == 'girl' or subject_id == 'woman':
        prompt_text = prompt_text.replace('his/her', 'her')
    print(f'Prompt: {prompt_text}')
    # 將每一筆資料結構化，方便後續追蹤與評估對齊度
    generated_data.append({
        "id": subject_id,
        # "view_condition": view,
        # "age": age,
        "pose_condition": pose,
        # "gaze_condition": gaze,
        "prompt": prompt_text
    })

# ==========================================
# 3. 輸出預覽與儲存檔案
# ==========================================
total_prompts = len(generated_data)
print(f"✅ 成功生成了 {total_prompts} 個不同的 Prompt 組合。\n")

print("🔍 前 5 個生成範例預覽：")
for data in generated_data[:5]:
    print(f"- {data['prompt']}")

filepath = "./posePrompt"
os.makedirs(filepath, exist_ok=True)
filename = "t2i_pose_prompts.jsonl"
# 將結果儲存為 JSONL 檔案
output_filename = os.path.join(filepath, filename)
with open(output_filename, "w", encoding="utf-8") as f:
    for data in generated_data:
        f.write(json.dumps(data) + "\n")

print(f"\n📁 所有資料已結構化並儲存至：{output_filename}")