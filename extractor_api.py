import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoTokenizer

# 假设原有的 SBASC 模型类在 models.SBASC.model 中
from models.SBASC.model import BERTLinear as SBASC_Model


class YelpFeatureExtractor:
    """
    工业级无状态情感抽取引擎 (Stateless Sentiment Extraction API)
    专为推荐系统 (SID) 设计，输出包含信息熵的 Soft Probability。
    """

    def __init__(self, domain_config_name="yelp_restaurant", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 初始化抽取引擎，设备: {self.device}")

        # 1. 加载配置 (这里简化了 Hydra 的读取，直接读取我们在上一阶段改好的 yaml)
        # 实际项目中可以使用 omegaconf 直接加载配置
        config_path = f"conf/domain/{domain_config_name}.yaml"
        self.cfg = OmegaConf.load(config_path)

        # 2. 挂载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.bert_mapper)

        # 3. 实例化模型并加载我们在 run.py 中强制固化的最优权重
        self.aspects = self.cfg.aspect_category_mapper
        self.polarities = self.cfg.sentiment_category_mapper

        # 【修复】：动态计算维度数量，并将其作为参数传给模型
        num_cat = len(self.aspects)     # 计算出 4
        num_pol = len(self.polarities)  # 计算出 2
        # 【修复】：只把 BERT 的绝对路径传给模型，而不是传整个配置字典
        self.model = SBASC_Model(self.cfg.bert_mapper, num_cat, num_pol).to(self.device)

        # 【关键】：加载模型权重 (确保您的 trainer.py 跑完后存了这个文件)
        model_weight_path = "model_final.pth"
        try:
            self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
            print("工业级权重加载成功！")
        except FileNotFoundError:
            print("未找到 model_final.pth，当前使用随机初始化权重（仅供调试）。")

        # 【工业级护城河】：永久冻结计算图，关闭 Dropout，极大提升推理速度与稳定性
        self.model.eval()

    @torch.no_grad()  # 强制不计算梯度，节省 50% 以上显存
    def extract_sentences_soft(self, sentences):
        """
        核心水泵：接收一批句子，吐出软概率
        """
        if not sentences:
            return []

        # 1. 防崩溃 Tokenize，加入截断和 Padding
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=128,  # 截断超长废话，防止 OOM
            return_tensors="pt"
        ).to(self.device)

        # 2. 模型前向传播 (获取 logits)
        # 假设模型返回 aspect_logits (N, num_aspects) 和 polarity_logits (N, num_polarities)
        #aspect_logits, polarity_logits = self.model(inputs['input_ids'], inputs['attention_mask'])
        # 2. 模型前向传播 (获取 logits)
        # 【修复】：原模型的 forward 强制要求输入真实标签计算 Loss，推理时我们用占位符 (Dummy) 骗过它
        batch_sz = inputs['input_ids'].size(0)
        dummy_labels_cat = torch.zeros(batch_sz, dtype=torch.long).to(self.device)
        dummy_labels_pol = torch.zeros(batch_sz, dtype=torch.long).to(self.device)
        
        # 接收返回值时，用 _ 忽略掉没用的 dummy loss
        _, aspect_logits, polarity_logits = self.model(
            labels_cat=dummy_labels_cat, 
            labels_pol=dummy_labels_pol, 
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask']
        )


        # ==========================================
        # 3. 【改造核心】：从 Hard Label 转向 Soft Probability
        # ==========================================
        # 使用 Sigmoid 激活 Aspect（因为一句话可能同时提到食物和服务，属于多标签）
        aspect_probs = torch.sigmoid(aspect_logits).cpu().numpy()

        # 使用 Softmax 激活 Polarity（因为正负极性是互斥的，属于多分类）
        polarity_probs = F.softmax(polarity_logits, dim=1).cpu().numpy()

        # 4. 组装并返回友好的字典格式
        batch_results = []
        for i in range(len(sentences)):
            asp_dict = {self.aspects[j]: float(aspect_probs[i][j]) for j in range(len(self.aspects))}
            pol_dict = {self.polarities[j]: float(polarity_probs[i][j]) for j in range(len(self.polarities))}

            batch_results.append({
                "sentence": sentences[i],
                "aspects": asp_dict,
                "polarities": pol_dict
            })

        return batch_results