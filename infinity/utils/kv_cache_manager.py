"""
KVCacheManager: 管理 Infinity Transformer 的 self-attention KV cache 快照。

用途：
  在 KV-Edit 交錯式管線中，三個 phase（source gen、phase 1.7、target gen）
  共用同一個模型，但各自維護獨立的 KV cache 歷史。
  本類別提供 save / restore / clear 操作，讓外部管線可以在切換 phase 時
  快速切換 KV cache 狀態。

同時管理 generation 中間狀態（summed_codes、last_stage 等），
使得 scale-by-scale 的暫停/恢復成為可能。
"""

import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class ScaleGenState:
    """保存單一 phase 在某個 scale 結束後的完整生成狀態。"""
    summed_codes: Any = 0          # torch.Tensor or int(0)
    last_stage: Optional[torch.Tensor] = None
    cur_L: int = 0
    idx_Bld_list: List[torch.Tensor] = field(default_factory=list)
    last_completed_scale: int = -1  # 最後完成的 scale index


class KVCacheManager:
    """管理多個 phase 的 self-attention KV cache 快照。

    每個 phase（例如 'source', 'phase17', 'target'）擁有獨立的 KV cache。
    在切換 phase 前先 save，切換後 restore，即可實現交錯式處理。
    """

    def __init__(self):
        self._snapshots: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
        self._gen_states: Dict[str, ScaleGenState] = {}

    # ── KV Cache 快照 ──────────────────────────────────────

    def save_kv_cache(self, model, phase_name: str, offload_to_cpu: bool = False) -> None:
        """從模型所有 SelfAttention block 取出 KV cache 並儲存。

        Args:
            offload_to_cpu: True 時快照存到 CPU，節省 GPU 記憶體。
                            適用於 si >= kv_blend_scales 時不需要 GPU 上做 blend 的場景。
        """
        from infinity.models.basic import CrossAttnBlock
        cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        blocks = model.unregistered_blocks
        for idx, block in enumerate(blocks):
            sa = block.sa if isinstance(block, CrossAttnBlock) else block.attn
            if sa.cached_k is not None:
                k, v = sa.cached_k.clone(), sa.cached_v.clone()
                if offload_to_cpu:
                    k, v = k.cpu(), v.cpu()
                cache[idx] = (k, v)
        self._snapshots[phase_name] = cache

    def restore_kv_cache(self, model, phase_name: str) -> None:
        """將先前儲存的 KV cache 寫回模型。若該 phase 尚無快照則清空。
        自動處理 CPU→GPU 搬移（若快照被 offload 到 CPU）。
        """
        from infinity.models.basic import CrossAttnBlock
        cache = self._snapshots.get(phase_name, {})
        blocks = model.unregistered_blocks
        for idx, block in enumerate(blocks):
            sa = block.sa if isinstance(block, CrossAttnBlock) else block.attn
            if idx in cache:
                k, v = cache[idx]
                device = next(model.parameters()).device
                sa.cached_k = k.clone().to(device)
                sa.cached_v = v.clone().to(device)
            else:
                sa.cached_k = None
                sa.cached_v = None

    def clear_kv_cache(self, model) -> None:
        """清空模型所有 SelfAttention block 的 KV cache（不影響快照）。"""
        from infinity.models.basic import CrossAttnBlock
        for block in model.unregistered_blocks:
            sa = block.sa if isinstance(block, CrossAttnBlock) else block.attn
            sa.cached_k = None
            sa.cached_v = None

    def delete_snapshot(self, phase_name: str) -> None:
        """刪除指定 phase 的快照以釋放記憶體。"""
        self._snapshots.pop(phase_name, None)

    def clear_all(self) -> None:
        """清空所有快照和 generation 狀態。"""
        for snap in self._snapshots.values():
            for k, v in snap.items():
                del k, v
        self._snapshots.clear()
        self._gen_states.clear()

    # ── Generation 狀態管理 ────────────────────────────────

    def save_gen_state(
        self,
        phase_name: str,
        summed_codes,
        last_stage: Optional[torch.Tensor],
        cur_L: int,
        idx_Bld_list: List[torch.Tensor],
        last_completed_scale: int,
    ) -> None:
        """儲存某個 phase 的 generation 中間狀態。"""
        state = ScaleGenState(
            summed_codes=(
                summed_codes.clone()
                if isinstance(summed_codes, torch.Tensor)
                else summed_codes
            ),
            last_stage=last_stage.clone() if last_stage is not None else None,
            cur_L=cur_L,
            idx_Bld_list=[t.clone() for t in idx_Bld_list],
            last_completed_scale=last_completed_scale,
        )
        self._gen_states[phase_name] = state

    def load_gen_state(self, phase_name: str) -> Optional[ScaleGenState]:
        """載入某個 phase 的 generation 中間狀態（clone 後回傳）。"""
        state = self._gen_states.get(phase_name)
        if state is None:
            return None
        return ScaleGenState(
            summed_codes=(
                state.summed_codes.clone()
                if isinstance(state.summed_codes, torch.Tensor)
                else state.summed_codes
            ),
            last_stage=state.last_stage.clone() if state.last_stage is not None else None,
            cur_L=state.cur_L,
            idx_Bld_list=[t.clone() for t in state.idx_Bld_list],
            last_completed_scale=state.last_completed_scale,
        )

    def has_state(self, phase_name: str) -> bool:
        return phase_name in self._gen_states

    # ── 進階：從 source KV cache 注入結構資訊到 target ──

    def inject_source_kv_to_target(
        self,
        model,
        source_phase: str,
        target_phase: str,
        mask: Optional[torch.Tensor] = None,
        blend_ratio: float = 0.0,
    ) -> None:
        """
        將 source phase 的 KV cache 混合到 target phase。

        Args:
            model: Infinity 模型
            source_phase: 來源 phase 名稱
            target_phase: 目標 phase 名稱
            mask: 空間遮罩 [1, L, 1]，True = 使用 source KV
            blend_ratio: 0.0 = 完全 target, 1.0 = 完全 source
        """
        from infinity.models.basic import CrossAttnBlock
        src_cache = self._snapshots.get(source_phase, {})
        tgt_cache = self._snapshots.get(target_phase, {})
        if not src_cache:
            return

        blended: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for idx in src_cache:
            src_k, src_v = src_cache[idx]
            if idx in tgt_cache:
                tgt_k, tgt_v = tgt_cache[idx]
                if blend_ratio >= 1.0:
                    blended[idx] = (src_k.clone(), src_v.clone())
                elif blend_ratio <= 0.0:
                    blended[idx] = (tgt_k.clone(), tgt_v.clone())
                else:
                    # source 可能比 target 多一個 scale 的 token（交錯管線中
                    # source 剛完成當前 scale，target 還在前一個 scale 的狀態）
                    # 找出 seq_len 所在的維度，只 blend 共同前綴
                    L_dim = None
                    for d in range(src_k.ndim):
                        if src_k.shape[d] != tgt_k.shape[d]:
                            L_dim = d
                            break

                    if L_dim is not None:
                        min_L = min(src_k.shape[L_dim], tgt_k.shape[L_dim])
                        src_k_prefix = src_k.narrow(L_dim, 0, min_L)
                        src_v_prefix = src_v.narrow(L_dim, 0, min_L)
                        tgt_k_prefix = tgt_k.narrow(L_dim, 0, min_L)
                        tgt_v_prefix = tgt_v.narrow(L_dim, 0, min_L)
                        bk = tgt_k_prefix * (1 - blend_ratio) + src_k_prefix * blend_ratio
                        bv = tgt_v_prefix * (1 - blend_ratio) + src_v_prefix * blend_ratio
                        # 保留 target 多出的 token（若有）
                        if tgt_k.shape[L_dim] > min_L:
                            rest_k = tgt_k.narrow(L_dim, min_L, tgt_k.shape[L_dim] - min_L)
                            rest_v = tgt_v.narrow(L_dim, min_L, tgt_v.shape[L_dim] - min_L)
                            bk = torch.cat([bk, rest_k], dim=L_dim)
                            bv = torch.cat([bv, rest_v], dim=L_dim)
                    else:
                        # shape 完全一致，直接 blend
                        bk = tgt_k * (1 - blend_ratio) + src_k * blend_ratio
                        bv = tgt_v * (1 - blend_ratio) + src_v * blend_ratio
                    blended[idx] = (bk, bv)
            else:
                blended[idx] = (src_k.clone(), src_v.clone())

        self._snapshots[target_phase] = blended

    def __repr__(self) -> str:
        phases = list(self._snapshots.keys())
        states = list(self._gen_states.keys())
        return (
            f"KVCacheManager(snapshots={phases}, gen_states={states})"
        )
