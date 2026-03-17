import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.nn.utils as utils
from torch.optim.lr_scheduler import CosineAnnealingLR

from params import HYPER_PARAMS, calculate_combined_reward

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化我们之前定义的参数化模型
# channels=256 对应你要求的专家意见层数
# model = MahjongDecisionNet(channels=256).to(torch.device)

def train_loop(model, loader, optimizer, device):
    # 1. 初始化半精度缩放器 (AMP)：
    # 作用：将 Float32 自动降为 Float16 计算，显存减半，且由于 Float16 范围小，
    # Scaler 会动态放大 Loss 防止数值变 0。
    scaler = GradScaler('cuda')
    
    # 2. 学习率调度器：使用余弦退火算法
    # 作用：随着训练进行，学习率像余弦曲线一样下降，最后阶段极其微小，方便模型“收针”寻找全局最优解。
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
    # 3. 损失函数：交叉熵
    # reduction='none' 极度重要！它不直接求平均，而是返回每个样本的 Loss。
    # 这样我们才能给每一手牌单独乘以我们上面算出来的“奖惩权重”。
    criterion = nn.CrossEntropyLoss(reduction='none')

    model.train() # 开启训练模式（激活 BatchNorm）
    
    for batch in loader:
        # 将数据搬运到 GPU (device)
        u_in, tiles, target, f_rank, s_change, r_imp, scores = [b.to(device) for b in batch]
        optimizer.zero_grad() # 清空上一轮的梯度
        
        # --- 核心训练块 ---
        with autocast('cuda'): # 开启自动混合精度上下文
            # a. 前向传播：计算模型预测结果
            output = model(u_in, tiles).squeeze(-1) # 得到 (Batch, 37) 的打牌概率分布
            
            # b. 计算原始 Loss：对比预测值与玩家实际打出的牌
            raw_loss = criterion(output, target)
            
            # c. 注入“灵魂”：计算本 Batch 每一个样本的动态奖惩权重
            weights = []
            for i in range(len(f_rank)):
                w = calculate_combined_reward(f_rank[i].item(), s_change[i].item(), r_imp[i].item(), scores[i], HYPER_PARAMS)
                weights.append(w)
            weight_tensor = torch.tensor(weights).to(device)
            
            # d. 加权损失：将 Loss 乘以权重再求平均
            # 重要动作（权重高）对梯度贡献大；错误动作（权重负）会让梯度反向。
            loss = (raw_loss * weight_tensor).mean()

        # --- 反向传播与稳定性优化 ---
        # 1. 缩放 Loss 并反向传播梯度
        scaler.scale(loss).backward()
        
        # 2. 梯度裁剪 (Gradient Clipping)：
        # 在更新参数前，强行将梯度的模长限制在 1.0 以内。
        # 即使 Loss 很大，参数更新的步长也不会失控，有效防止 50 层网络在训练初期爆炸。
        scaler.unscale_(optimizer) # 在裁剪前必须先取消缩放
        utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 3. 更新模型参数
        scaler.step(optimizer)
        scaler.update() # 更新 Scaler 的缩放因子
    
    # 每个 Epoch 结束，调整一次学习率
    scheduler.step()