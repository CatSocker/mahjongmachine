import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 第一部分：定义参数化残差块 ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # 严格执行 3x1 卷积，通道数由参数 channels 决定
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity 
        return F.relu(out)
    

# --- 第二部分：主模型 (专家数量参数化) ---
class MahjongDecisionNet(nn.Module):
    def __init__(self, channels=256, cycles=10):
        """
        :param channels: 专家意见层的数量 (即卷积通道数)，默认为 256
        :param cycles: 残差循环的次数，默认为 10
        """
        super(MahjongDecisionNet, self).__init__()
        self.expert_channels = channels
        self.cycles = cycles

        # --- [1. 上路: 局势流扩展至 1024 维] ---
        # 这里有一个点，就是当channels=256这个参数变化时，我们希望上路的维度仍然是可控的，扩展次数也是可变的。这一部分代码需要优化。
        self.upper_branch = nn.Sequential(
            nn.Linear(63, 128),    nn.BatchNorm1d(128),  nn.ReLU(),
            nn.Linear(128, 256),   nn.BatchNorm1d(256),  nn.ReLU(),
            nn.Linear(256, 512),  nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU()
        )
        # 如果训练时发现模型对某些关键局势（如点炮后的惩罚）反应不够灵敏，可以在训练脚本里给上路特征加入少量的 Dropout(0.1)。
        
        # --- [2. 下路: 牌面流] ---
        # 初始卷积：将 1 通道提升至自定义的 channels 数量
        self.conv_init = nn.Conv2d(1, self.expert_channels, kernel_size=(1, 3), padding=(0, 1))
        self.bn_init = nn.BatchNorm2d(self.expert_channels)
        
        # 若干次残差循环 (Suphx 级深度)，使用参数化通道
        self.res_layers = nn.Sequential(*[ResidualBlock(self.expert_channels) for _ in range(self.cycles)])
        
        # --- 修正后的 2c 步骤 (专家保留卷积) ---
        # kernel_size=(125, 1) 负责将 125 个特征层压缩为 1
        # 输入通道和输出通道均由 self.expert_channels 决定
        self.conv_expert = nn.Conv2d(self.expert_channels, self.expert_channels, kernel_size=(125, 1)) 
        self.bn_expert = nn.BatchNorm2d(self.expert_channels)
        
        # 2d. Flatten
        # 此时尺寸为: channels * 1(H) * 37(W)
        self.flatten = nn.Flatten()
        
        # --- [3. 融合与输出] ---
        # 下路维度计算: channels * 37
        down_stream_dim = self.expert_channels * 37
        
        # 融合后的全连接层: 上路 (1024) + 下路 (channels * 37)
        self.pre_output = nn.Linear(1024 + down_stream_dim, 37)

    def forward(self, u_bools, u_floats, tiles_4d):
        """
        tiles_4d: (Batch, 125, 37, 1)
        """
        # 步骤 1: 上路处理 (局势)
        u_in = torch.cat((u_bools, u_floats), dim=1)
        x_up = self.upper_branch(u_in) # (B, 1024)
        
        # 步骤 2: 下路处理 (牌面)
        x_down = tiles_4d.permute(0, 3, 1, 2).float() # 调整维度为 (B, 1, 125, 37)
        
        # 卷积提取与残差深度学习
        x_down = F.relu(self.bn_init(self.conv_init(x_down)))
        x_down = self.res_layers(x_down) # 形状: (B, channels, 125, 37)
        
        # 2c 修正：执行纵向压缩，保留专家通道
        x_down = F.relu(self.bn_expert(self.conv_expert(x_down))) # 形状: (B, channels, 1, 37)
        
        # 2d 展平
        x_down_flat = self.flatten(x_down) # 形状: (B, channels * 37)
        
        # 步骤 3: 融合拼合
        combined = torch.cat((x_up, x_down_flat), dim=1)
        
        # 步骤 4: 映射与 Mask
        logits = self.pre_output(combined).unsqueeze(-1) # (B, 37, 1)
        action_mask = tiles_4d[:, 0, :, :]               # (B, 37, 1)
        
        return logits * action_mask

# --- 鸣牌/立直决策模型 (专家数量参数化) ---
class MahjongBoolDecisionNet(nn.Module):
    def __init__(self, channels=256, cycles=10):
        """
        :param channels: 专家意见层的数量 (即卷积通道数)，默认为 256
        :param cycles: 残差循环的次数，默认为 10
        """
        super(MahjongBoolDecisionNet, self).__init__()
        self.expert_channels = channels
        self.cycles = cycles

        # --- [1. 上路: 局势流扩展至 1024 维] ---
        # 这里有一个点，就是当channels=256这个参数变化时，我们希望上路的维度仍然是可控的，扩展次数也是可变的。这一部分代码需要优化。
        self.upper_branch = nn.Sequential(
            nn.Linear(63, 128),    nn.BatchNorm1d(128),  nn.ReLU(),
            nn.Linear(128, 256),   nn.BatchNorm1d(256),  nn.ReLU(),
            nn.Linear(256, 512),  nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU()
        )
        # 如果训练时发现模型对某些关键局势（如点炮后的惩罚）反应不够灵敏，可以在训练脚本里给上路特征加入少量的 Dropout(0.1)。
        
        # --- [2. 下路: 牌面流] ---
        # 初始卷积：将 1 通道提升至自定义的 channels 数量
        self.conv_init = nn.Conv2d(1, self.expert_channels, kernel_size=(1, 3), padding=(0, 1))
        self.bn_init = nn.BatchNorm2d(self.expert_channels)
        
        # 若干次残差循环 (Suphx 级深度)，使用参数化通道
        self.res_layers = nn.Sequential(*[ResidualBlock(self.expert_channels) for _ in range(self.cycles)])

        # --- 修正后的 2c 步骤 (专家保留卷积) ---
        # kernel_size=(125, 1) 负责将 125 个特征层压缩为 1
        # 输入通道和输出通道均由 self.expert_channels 决定
        self.conv_expert = nn.Conv2d(self.expert_channels, self.expert_channels, kernel_size=(125, 1)) 
        self.bn_expert = nn.BatchNorm2d(self.expert_channels)

        # 2d. Flatten
        # 此时尺寸为: channels * 1(H) * 37(W)
        self.flatten = nn.Flatten()

        # --- [1.5 中路：决策之后的牌型] ---
        self.conv_init_naku = nn.Conv2d(1, 10, kernel_size=(1, 3), padding=(0, 1))
        self.bn_init_naku = nn.BatchNorm2d(10)
        self.res_layers_naku = nn.Sequential(*[ResidualBlock(10) for _ in range(10)])
        self.conv_expert_naku = nn.Conv2d(10, 10, kernel_size=(8, 1)) 
        self.bn_expert_naku = nn.BatchNorm2d(10)
        
        # --- [3. 融合与输出] ---
        # 下路维度计算: channels * 37
        down_stream_dim = self.expert_channels * 37
        
        # 融合后的全连接层: 上路 (1024) + 下路 (channels * 37)
        self.pre_output = nn.Linear(1024 + down_stream_dim, 1)

    def forward(self, u_bools, u_floats, naku_target, tiles_4d):
        """
        tiles_4d: (Batch, 125, 37, 1)
        """
        # 步骤 1: 上路处理 (局势)
        u_in = torch.cat((u_bools, u_floats), dim=1)
        x_up = self.upper_branch(u_in) # (B, 1024)

        # 步骤 1.5: 中路处理 (决策后的牌型)

        x_naku = naku_target.permute(0, 3, 1, 2).float() # # 调整维度为 (B, 1, 8, 37)
        
        # 卷积提取与残差深度学习
        x_naku = F.relu(self.bn_init_naku(self.conv_init_naku(x_naku)))
        x_naku = self.res_layers_naku(x_naku) # 形状: (B, 10, 8, 37)
        
        # 修正：执行纵向压缩，保留专家通道
        x_naku = F.relu(self.bn_expert_naku(self.conv_expert_naku(x_naku))) # 形状: (B, 10, 1, 37)

        # 展平
        x_naku_flat = self.flatten(x_naku) # 形状: (B, 10 * 37)

        # 步骤 2: 下路处理 (牌面)
        x_down = tiles_4d.permute(0, 3, 1, 2).float() # 调整维度为 (B, 1, 125, 37)
        
        # 卷积提取与残差深度学习
        x_down = F.relu(self.bn_init(self.conv_init(x_down)))
        x_down = self.res_layers(x_down) # 形状: (B, channels, 125, 37)
        
        # 2c 修正：执行纵向压缩，保留专家通道
        x_down = F.relu(self.bn_expert(self.conv_expert(x_down))) # 形状: (B, channels, 1, 37)
        
        # 2d 展平
        x_down_flat = self.flatten(x_down) # 形状: (B, channels * 37)
        
        # 步骤 3: 融合拼合
        combined = torch.cat((x_up, x_naku_flat, x_down_flat), dim=1)

        # 步骤 4: 映射
        logit_bool = self.pre_output(combined) # (B, 1)
        
        return logit_bool


class RewardPredictor(nn.Module):
    def __init__(self, channels=128, cycles=10):
        super().__init__()
        # 简化版 ResNet，用于提取当前局势特征
        self.conv_init = nn.Conv2d(1, channels, kernel_size=(1, 3), padding=(0, 1))
        self.res_layers = nn.Sequential(*[ResidualBlock(channels) for _ in range(cycles)])
        self.flatten = nn.Flatten()
        # 输入：卷积特征 + 59维全局特征
        # 输出：4个数值 (对应四家局末的 Score Change)
        self.fc = nn.Sequential(
            nn.Linear(channels * 37 + 63, 512),
            nn.ReLU(),
            nn.Linear(512, 4) 
        )

    def forward(self, u_in, tiles_4d): # u_in 已经是拼合后的全局特征 (B, 63)，tiles_4d 是牌面特征 (B, 125, 37, 1)
        x = tiles_4d.permute(0, 3, 1, 2).float()
        x = F.relu(self.conv_init(x))
        x = self.res_layers(x)
        x_flat = self.flatten(x[:, :, -1, :]) # 取最后一层特征
        combined = torch.cat((x_flat, u_in), dim=1)
        return self.fc(combined)

# --- 实例检查 ---

for channels, layers in [(32, 10), (64, 10), (128, 10), (256, 10), (512, 10), (32, 50), (64, 50), (128, 50), (256, 50), (512, 50)]:
    model = MahjongDecisionNet(channels=channels, cycles=layers)
    # model = MahjongBoolDecisionNet(channels=channels, cycles=layers)
    print(f"专家层数: {channels}, 残差层数: {layers}, 模型参数总量: {sum(p.numel() for p in model.parameters()):,}")

#   专家层数    10x模型参数     50x模型参数
#   32          975,333         1,228,773
#   64          1,589,669       2,588,069
#   128         3,954,981       7,917,861
#   256         13,232,165      29,022,245
#   512         49,972,773      113,010,213

