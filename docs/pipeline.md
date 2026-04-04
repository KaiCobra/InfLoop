phase 1 主要是需要 source focus text 對應的attention 區域  --> $mask_s$ (這邊會取source image需要保留的token, 使用attention 最低的做為保留)
(input: $I_s$, $P_s$
output: $Mask_s$)

phase 2 主要是需要 target focus text 對應的attention 區域 --> $mask_t$
(input: $P_t$, (有沒有 $I_s$ + $Mask_s$ 我要確認一下)
output: $Mask_t$)

phase 3 的話 就是將 $mask_s$ & $mask_t$ 蓋在source image 上面 然後讓phase 3 產生
(input: $Mask_s$+$Mask_t$+$I_s$, $P_t$
output: $I_t$)

===========
source focus word & target focus word