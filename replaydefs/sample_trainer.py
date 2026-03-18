import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import torch.nn.utils as utils
from torch.optim.lr_scheduler import CosineAnnealingLR
# 假设计算逻辑已经解耦到 params.py 中
from params import HYPER_PARAMS, calculate_combined_reward

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MahjongDecisionNet(channels=256).to(torch.device)

def train_loop(model, reward_predictor, loader, optimizer, device, current_alpha, reward_alpha=0.25):
    """
    更新后的训练循环
    :param current_alpha: 论文中的 alpha 参数，用于平衡交叉熵与策略熵
    :param reward_predictor: Suphx 3.3 中的全局奖励预测器模型
    :param reward_alpha: 真实终局得分的权重比例，预测得分权重自动计算为 (1 - reward_alpha)
    """
    # 1. 初始化半精度缩放器 (AMP)
    # 针对 50 层残差模型，混合精度能有效防止梯度在深层传播时出现数值下溢
    scaler = GradScaler('cuda')
    
    # 2. 学习率调度器：余弦退火
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
    # 3. 损失函数：设置为 none，以便手动应用顺位/点数奖惩权重
    criterion = nn.CrossEntropyLoss(reduction='none')

    model.train() 
    # 预测器作为教师模型，通常保持评估模式
    reward_predictor.eval()
    
    total_epoch_entropy = 0.0 # 累计熵值，用于返回给外层更新 alpha
    
    for batch in loader:
        # 搬运数据到指定设备
        u_in, tiles, target, f_rank, s_change, r_imp, scores = [b.to(device) for b in batch]
        optimizer.zero_grad() 
        
        # --- 核心训练块 ---
        with autocast('cuda'): 
            # a. 前向传播：得到未归一化的概率分布 (Logits)
            logits = model(u_in, tiles).squeeze(-1) # (Batch, 37)
            
            # b. 计算策略熵 (Policy Entropy) [Suphx 强化学习关键点]
            # 计算 softmax 概率 p 和 log 概率 log_p
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # H = -Σ p * log(p) 
            # 熵越大代表 AI 动作越多样（敢于探索），熵越小代表 AI 越武断
            batch_entropy = -(probs * log_probs).sum(dim=-1).mean()
            total_epoch_entropy += batch_entropy.item()
            
            # --- [新集成：Suphx 3.3 全局奖励预测逻辑] ---
            with torch.no_grad():
                # 预测当前局势下四家的预期终局得分变动
                # pred_scores 形状为 (Batch, 4)，我们取当前玩家(通常是Batch中的主视角)的预测值
                pred_scores = reward_predictor(u_in, tiles)
                pred_s_change = pred_scores[:, 0] # 假设预测器输出的第0位是当前玩家
            
            # 融合奖励：将“实际终局得分”与“当前时刻预测得分”进行加权
            # 使用函数变量 reward_alpha 平衡真实结果与当前动作的即时价值预测
            mixed_s_change = reward_alpha * s_change + (1.0 - reward_alpha) * pred_s_change
            
            # c. 计算基础加权损失 (基于玩家行为的监督/强化学习)
            raw_loss = criterion(logits, target)
            
            # 根据顺位、混合后的点数差、即时排名变化计算每个样本的灵魂权重
            weights = []
            for i in range(len(f_rank)):
                # 使用 mixed_s_change 替代原有的 s_change，提供更及时的信用分配
                w = calculate_combined_reward(f_rank[i].item(), mixed_s_change[i].item(), r_imp[i].item(), scores[i], HYPER_PARAMS)
                weights.append(w)
            weight_tensor = torch.tensor(weights).to(device)
            
            # 合并基础损失
            weighted_ce_loss = (raw_loss * weight_tensor).mean()

            # d. 最终总损失 [对应 Suphx 论文公式 (2)]
            # Loss = 行为损失 - alpha * 策略熵
            # 减号是因为 Loss 是要减小的，减去熵意味着我们在增大熵（最大化探索性）
            loss = weighted_ce_loss - current_alpha * batch_entropy

        # --- 反向传播与稳定性优化 ---
        # 1. 缩放并反向传播
        scaler.scale(loss).backward()
        
        # 2. 梯度裁剪：保护 50 层网络不被极端点炮样本的巨大梯度击穿
        scaler.unscale_(optimizer)
        utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 3. 参数更新
        scaler.step(optimizer)
        scaler.update() 
    
    scheduler.step()
    
    # 计算当前 Epoch 的平均熵值
    avg_entropy = total_epoch_entropy / len(loader)
    return avg_entropy