# Scale-wise Rollback Mechanism for Infinity Model

## 概述 (Overview)

這個功能允許 Infinity 模型在 inference 過程中對特定的 scale 進行回退和重新生成，實現自我修正機制。

## 運作原理 (How It Works)

### 正常生成流程
```
Scale: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> ... -> 13
```

### 使用回退機制（例如在 scale 4 回退 1 次）
```
Scale: 1 -> 2 -> 3 -> 4 -> [回到 3] -> 4 (重新生成) -> 5 -> 6 -> ... -> 13
```

### 流程說明
1. **正常生成**: 模型按照 scale_schedule 順序生成 scale 1, 2, 3
2. **到達 scale 4**: 模型在生成 scale 4 之前保存當前狀態
3. **生成 scale 4**: 第一次生成 scale 4
4. **觸發回退**: 檢測到 scale 4 需要回退
5. **恢復狀態**: 恢復到 scale 3 完成時的狀態
6. **重新生成**: 重新生成 scale 4（使用不同的隨機採樣）
7. **繼續**: 繼續生成後續的 scale 5, 6, ..., 13

## 使用方法 (Usage)

### 基本用法

```python
# 定義回退配置：key 是 scale 索引，value 是重試次數
rollback_schedule = {
    4: 1  # 在 scale index 4 回退 1 次（注意：索引從 0 開始）
}

# 在推理時傳入 rollback_schedule
ret, idx_Bl_list, img = model.autoregressive_infer_cfg(
    vae=vae,
    label_B_or_BLT=text_cond,
    scale_schedule=scale_schedule,
    cfg_list=cfg_list,
    tau_list=tau_list,
    rollback_schedule=rollback_schedule,  # 啟用回退機制
    ret_img=True,
    B=1,
)
```

### 配置示例

#### 1. 單一 Scale 回退
```python
# Scale 4 生成後回退並重新生成 1 次
rollback_schedule = {4: 1}
```
執行順序: `0 -> 1 -> 2 -> 3 -> 4 -> [back] -> 4 -> 5 -> ...`

#### 2. 多次重試
```python
# Scale 6 重新生成 3 次
rollback_schedule = {6: 3}
```
執行順序: `... -> 5 -> 6 -> [back] -> 6 -> [back] -> 6 -> [back] -> 6 -> 7 -> ...`

#### 3. 多個 Scale 回退
```python
# 在多個 scale 設置回退
rollback_schedule = {
    3: 1,  # Scale 3 回退 1 次
    7: 2,  # Scale 7 回退 2 次
    10: 1, # Scale 10 回退 1 次
}
```

#### 4. 早期階段精煉
```python
# 專注於精煉早期 scale 以獲得更好的基礎
rollback_schedule = {
    2: 1,
    3: 1,
    4: 2,  # 關鍵 scale 多次精煉
}
```

## 實現細節 (Implementation Details)

### 保存的狀態
在每個需要回退的 scale 之前，系統會保存：
- `last_stage`: 隱藏狀態
- `accu_BChw`: 累積的特徵
- `summed_codes`: 累積的 codes
- `cur_L`: 當前序列長度
- `ret`: 已生成的結果列表
- `idx_Bl_list`: 已生成的索引列表

### 回退過程
1. 生成目標 scale
2. 檢查是否需要回退
3. 如果需要，恢復之前保存的狀態
4. 使用新的隨機採樣重新生成該 scale
5. 重複直到達到指定的重試次數

### 隨機性
每次重新生成都會使用不同的隨機採樣（如果使用 top-k/top-p），因此每次生成的結果都會不同。這允許模型探索不同的可能性並選擇更好的結果。

## 應用場景 (Use Cases)

### 1. 自我修正
模型可以在發現當前生成不理想時回退重試

### 2. 探索式生成
在關鍵 scale 多次採樣以探索不同的生成路徑

### 3. 質量提升
通過多次嘗試提高特定 scale 的生成質量

### 4. 漸進式精煉
在早期 scale 進行多次精煉以建立更好的基礎

## 性能考慮 (Performance Considerations)

- **計算成本**: 每次回退都會重新計算該 scale，增加計算時間
- **內存使用**: 需要保存狀態快照，會增加內存使用
- **建議**: 謹慎選擇需要回退的 scale，避免過度使用

## 進階用法 (Advanced Usage)

### 條件式回退
未來可以擴展為基於生成質量的條件式回退：

```python
# 偽代碼示例
def conditional_rollback(generated_result, quality_threshold):
    quality_score = evaluate_quality(generated_result)
    if quality_score < quality_threshold:
        return True  # 需要回退
    return False
```

### 自適應回退
可以根據生成過程動態調整回退策略

## 調試信息 (Debug Information)

啟用回退時，會輸出以下信息：
```
[Rollback] Saving state before scale 4
[Rollback] Scale 4 completed. Rolling back to scale 3 (retry 1/1)
[Rollback] State restored. Regenerating scale 4...
```

## 參考 (References)

詳細示例請參考: `tools/rollback_example.py`

## 更新日誌 (Changelog)

- 2026-02-07: 初始實現 scale-wise rollback mechanism
