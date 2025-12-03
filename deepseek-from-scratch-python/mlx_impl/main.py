import mlx.core as mx
import mlx.nn as nn
import time
from attention import MultiQueryAttention, GroupedQueryAttention, MultiHeadLatentAttention
from moe import DeepSeekMoE
from mtp import MTPModel
from r1 import ReasoningModel
from grpo import GRPOTrainer
from quantization import FP8Linear
from pipeline import ScalingLaws, DataMixer, PipelineConfig, CurriculumScheduler, DistributedConfig
from sft import DeepSeekChatTemplate, SFTConfig, SFTTrainer
from reward_model import RewardModelSimple
from dpo import DPOConfig
from distillation import DistillationConfig, KDLossType, compute_distillation_loss

def run_demos():
    print("=== DeepSeek from Scratch (MLX) ===")
    mx.set_default_device(mx.gpu)
    
    # --- Chapter 1 ---
    print("\n--- Chapter 1: Multi-Query Attention (MQA) ---")
    mqa = MultiQueryAttention(d_model=512, num_heads=8)
    x = mx.random.normal((4, 64, 512))
    out = mqa(x)
    print(f"MQA Output: {out.shape}")
    
    print("\n--- Chapter 1: Grouped-Query Attention (GQA) ---")
    gqa = GroupedQueryAttention(d_model=512, num_heads=32, num_groups=4)
    out = gqa(x)
    print(f"GQA Output: {out.shape}")
    
    # --- Chapter 2 ---
    print("\n--- Chapter 2: Multi-Head Latent Attention (MLA) ---")
    mla = MultiHeadLatentAttention(d_model=512, num_heads=8, d_latent=128, d_rope=64)
    out = mla(x)
    print(f"MLA Output: {out.shape}")
    
    # --- Chapter 3 ---
    print("\n--- Chapter 3: Mixture-of-Experts (MoE) ---")
    moe = DeepSeekMoE(d_model=512, d_hidden=1024, num_experts=10, num_shared=2, num_routed=8, top_k=2)
    out = moe(x)
    print(f"DeepSeek MoE Output: {out.shape}")
    
    # --- Chapter 4 ---
    print("\n--- Chapter 4: Multi-Token Prediction (MTP) ---")
    mtp = MTPModel(vocab_size=1000, d_model=512, num_layers=2, k_predictions=1)
    input_ids = mx.random.randint(0, 1000, (4, 64))
    main_logits, future_logits = mtp(input_ids)
    print(f"MTP Main Logits: {main_logits.shape}")
    print(f"MTP Future Logits: {future_logits[0].shape}")
    
    print("\n--- Chapter 4: FP8 Quantization Simulation ---")
    fp8 = FP8Linear(512, 512)
    start = time.time()
    out = fp8(x)
    mx.eval(out) # Force evaluation
    print(f"FP8 Forward Time: {(time.time() - start)*1000:.2f}ms")
    print(f"FP8 Output: {out.shape}")
    
    # --- Chapter 5 ---
    print("\n--- Chapter 5: DeepSeek-R1 (Reasoning) ---")
    r1 = ReasoningModel(vocab_size=1000, d_model=512)
    output = r1.generate_with_reasoning("DeepSeek architecture")
    print(f"R1 Output:\n{output}")
    
    # --- Chapter 6 ---
    print("\n--- Chapter 6: GRPO ---")
    grpo = GRPOTrainer(beta=0.01)
    logits = mx.random.normal((4, 10, 100))
    ref_logits = mx.random.normal((4, 10, 100))
    input_ids = mx.random.randint(0, 100, (4, 10))
    rewards = mx.array([1.0, 0.5, -0.5, 0.0])
    
    loss = grpo.compute_loss(logits, input_ids, rewards, ref_logits)
    print(f"GRPO Loss: {loss.item():.4f}")
    
    # --- Chapter 7: Training Pipeline ---
    print("\n" + "="*60)
    print("DeepSeek Training Pipeline Demo")
    print("=" * 60 + "\n")

    # Scaling Laws
    print("1. Scaling Laws")
    print("-" * 40)
    scaling = ScalingLaws()
    
    print(f"Predicted loss (7B, 2T tokens): {scaling.predict_loss(7e9, 2e12):.4f}")
    print(f"Recommended LR for 7B: {scaling.recommended_lr(7e9):.2e}")
    
    optimal = scaling.optimal_config(1e23)
    print("Optimal config for 1e23 FLOPs:")
    print(f"  Parameters: {optimal['optimal_params']/1e9:.1f}B")
    print(f"  Tokens: {optimal['optimal_tokens']/1e12:.1f}T")
    
    # Data Mixing
    print("\n2. Data Mixing")
    print("-" * 40)
    mixing = DataMixer({
        "web": 0.60,
        "code": 0.20,
        "math": 0.10,
        "books": 0.05,
        "scientific": 0.05,
    })
    
    print("Sampling probabilities:")
    probs = sorted(mixing.get_probs().items(), key=lambda x: x[1], reverse=True)
    for domain, prob in probs:
        print(f"  {domain}: {prob*100:.1f}%")
    
    # Curriculum Learning
    print("\n4. Curriculum Learning")
    print("-" * 40)
    curriculum = CurriculumScheduler(512, 4096, 10000, 5000)
    
    print("Sequence length progression:")
    for step in [0, 2500, 5000, 7500, 10000, 15000]:
        print(f"  Step {step:5d}: {curriculum.get_seq_length(step)}")
    
    print("Difficulty weight progression:")
    for step in [0, 1000, 2500, 5000, 10000]:
        print(f"  Step {step:5d}: {curriculum.get_difficulty_weight(step):.2f}")
    
    # Pipeline Config
    print("\n5. Pipeline Configuration")
    print("-" * 40)
    config = PipelineConfig()
    print(f"Model size: {config.model_size/1e9:.1f}B")
    print(f"Max steps: {config.max_steps}")
    print(f"Learning rate: {config.learning_rate:.0e}")
    
    # Distributed Config
    print("\n6. Distributed Configuration")
    print("-" * 40)
    dist = DistributedConfig(world_size=8, dp_size=8)
    print(f"World size: {dist.world_size}")
    print(f"Data parallelism: {dist.dp_size}")
    print(f"ZeRO stage: {dist.zero_stage}")
    
    print("\n" + "=" * 60)
    print("Pipeline Demo Complete!")
    print("=" * 60 + "\n")
    
    # --- Chapter 8: Post-Training ---
    print("\n" + "="*60)
    print("SFT Demo")
    print("=" * 60 + "\n")
    
    # Chat template
    print("1. Chat Template")
    print("-" * 40)
    
    template = DeepSeekChatTemplate()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ]
    
    formatted = template.format_conversation(messages)
    print("Formatted conversation:")
    print(formatted)
    
    # LoRA config
    print("\n2. LoRA Configuration")
    print("-" * 40)
    
    config = SFTConfig()
    print(f"LoRA rank: {config.lora_r}")
    print(f"LoRA alpha: {config.lora_alpha}")
    print(f"Target modules: {config.lora_target_modules}")
    
    # NEFTune
    print("\n3. NEFTune Configuration")
    print("-" * 40)
    print(f"NEFTune enabled: {config.use_neftune}")
    print(f"NEFTune alpha: {config.neftune_alpha}")
    print("Noise scale = alpha / sqrt(seq_len)")
    print(f"For seq_len=1024: noise = {config.neftune_alpha / (1024 ** 0.5):.4f}")
    
    print("\n" + "=" * 60)
    print("SFT Demo Complete!")
    print("=" * 60 + "\n")
    
    print("\n" + "="*60)
    print("Reward Model Demo")
    print("=" * 60 + "\n")
    
    # Simple reward model
    print("\n2. Simple Reward Model")
    print("-" * 40)
    
    model = RewardModelSimple(
        vocab_size=32000,
        hidden_size=512,
        num_layers=4,
        num_heads=8,
    )
    
    # Count parameters
    # mx.utils is not in mx.core
    import mlx.utils
    params = sum(v.size for k, v in mlx.utils.tree_flatten(model.parameters()))
    print(f"Parameters: {params:,}")
    
    # Forward pass
    batch_size, seq_len = 2, 128
    input_ids = mx.random.randint(0, 32000, (batch_size, seq_len))
    # attention_mask not used in simple model
    
    rewards = model(input_ids)
    print(f"Reward shape: {rewards.shape}")
    print(f"Rewards: {rewards.tolist()}")
    
    print("\n" + "=" * 60)
    print("Reward Model Demo Complete!")
    print("=" * 60 + "\n")
    
    # --- Chapter 9: Knowledge Distillation ---
    print("\n" + "="*60)
    print("Knowledge Distillation Demo")
    print("="*60 + "\n")
    
    # Configuration
    print("\n1. Distillation Configuration")
    print("-" * 40)
    
    config = DistillationConfig()
    print(f"Temperature: {config.temperature}")
    print(f"Alpha: {config.alpha}")
    print(f"KD Loss Type: {config.kd_loss_type}")
    
    # KD Loss Types
    print("\n2. Knowledge Distillation Losses")
    print("-" * 40)
    
    batch_size, seq_len, vocab_size = 2, 10, 1000
    student_logits = mx.random.normal((batch_size, seq_len, vocab_size))
    teacher_logits = mx.random.normal((batch_size, seq_len, vocab_size))
    
    for loss_type in KDLossType:
        cfg = DistillationConfig(kd_loss_type=loss_type)
        loss = compute_distillation_loss(student_logits, teacher_logits, cfg)
        print(f"{loss_type.name}: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("Distillation Demo Complete!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_demos()
