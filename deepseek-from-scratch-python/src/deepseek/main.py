import torch
import time
from deepseek.model.attention import MultiQueryAttention, GroupedQueryAttention
from deepseek.model.mla import MultiHeadLatentAttention, DeepSeekAttention
from deepseek.model.moe import DeepSeekMoE, StandardMoE
from deepseek.model.mtp import MTPModel
from deepseek.model.quantization import FP8Linear
from deepseek.model.r1 import ReasoningModel
from deepseek.training.grpo import GRPOTrainer
from deepseek.training.pipeline import ScalingLaws, DataMixer, PipelineConfig, CurriculumScheduler, DistributedConfig
from deepseek.training.sft import DeepSeekChatTemplate, SFTConfig, SFTTrainer
from deepseek.model.reward_model import RewardConfig, RewardModelSimple
from deepseek.training.dpo import DPOConfig, DPOTrainer
from deepseek.training.distillation import DistillationConfig, KDLossType, compute_distillation_loss, combined_distillation_loss, HiddenDistillation, compute_layer_mapping, ProgressiveConfig, ProgressiveDistiller, TemperatureScheduler
import torch.nn.functional as F

def run_demos():
    print("=== DeepSeek from Scratch (Python) ===")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # --- Chapter 1 ---
    print("\n--- Chapter 1: Multi-Query Attention (MQA) ---")
    mqa = MultiQueryAttention(d_model=512, num_heads=8).to(device)
    x = torch.randn(4, 64, 512).to(device)
    out = mqa(x)
    print(f"MQA Output: {out.shape}")
    
    print("\n--- Chapter 1: Grouped-Query Attention (GQA) ---")
    gqa = GroupedQueryAttention(d_model=512, num_heads=32, num_groups=4).to(device)
    out = gqa(x)
    print(f"GQA Output: {out.shape}")
    
    # --- Chapter 2 ---
    print("\n--- Chapter 2: Multi-Head Latent Attention (MLA) ---")
    mla = MultiHeadLatentAttention(d_model=512, num_heads=8, d_latent=128, d_rope=64).to(device)
    out = mla(x)
    print(f"MLA Output: {out.shape}")
    
    # --- Chapter 3 ---
    print("\n--- Chapter 3: Mixture-of-Experts (MoE) ---")
    moe = DeepSeekMoE(d_model=512, d_hidden=1024, num_experts=10, num_shared=2, num_routed=8, top_k=2).to(device)
    out = moe(x)
    print(f"DeepSeek MoE Output: {out.shape}")
    
    # --- Chapter 4 ---
    print("\n--- Chapter 4: Multi-Token Prediction (MTP) ---")
    mtp = MTPModel(vocab_size=1000, d_model=512, num_layers=2, k_predictions=1).to(device)
    input_ids = torch.randint(0, 1000, (4, 64)).to(device)
    main_logits, future_logits = mtp(input_ids)
    print(f"MTP Main Logits: {main_logits.shape}")
    print(f"MTP Future Logits: {future_logits[0].shape}")
    
    print("\n--- Chapter 4: FP8 Quantization Simulation ---")
    fp8 = FP8Linear(512, 512).to(device)
    start = time.time()
    out = fp8(x)
    print(f"FP8 Forward Time: {(time.time() - start)*1000:.2f}ms")
    print(f"FP8 Output: {out.shape}")
    
    # --- Chapter 5 ---
    print("\n--- Chapter 5: DeepSeek-R1 (Reasoning) ---")
    r1 = ReasoningModel(vocab_size=1000, d_model=512).to(device)
    output = r1.generate_with_reasoning("DeepSeek architecture")
    print(f"R1 Output:\n{output}")
    
    # --- Chapter 6 ---
    print("\n--- Chapter 6: GRPO ---")
    grpo = GRPOTrainer(beta=0.01)
    logits = torch.randn(4, 10, 100).to(device)
    ref_logits = torch.randn(4, 10, 100).to(device)
    input_ids = torch.randint(0, 100, (4, 10)).to(device)
    rewards = torch.tensor([1.0, 0.5, -0.5, 0.0]).to(device)
    
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
    
    # WSD Scheduler
    print("\n3. WSD Learning Rate Schedule")
    print("-" * 40)
    config = PipelineConfig()
    # Create dummy optimizer for scheduler demo
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([dummy_param], lr=1e-4)
    scheduler = config.create_scheduler(optimizer)
    
    print("LR at different steps:")
    for step in [0, 1000, 2000, 50000, 82000, 100000]:
        print(f"  Step {step:6d}: {scheduler.get_lr(step):.2e}")
    
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
    print(f"Model size: {config.model_size/1e9:.1f}B")
    print(f"Max steps: {config.max_steps}")
    print(f"Learning rate: {config.learning_rate:.0e}")
    
    eff_batch_size = config.data.batch_size * config.gradient_accumulation_steps * config.distributed.world_size
    print(f"Effective batch size: {eff_batch_size}")
    
    tokens_per_step = eff_batch_size * config.data.seq_length
    print(f"Tokens per step: {tokens_per_step}")
    
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
    
    # Simulated LoRA savings
    original_params = 7_000_000_000
    lora_params = config.lora_r * (4096 + 4096) * len(config.lora_target_modules) * 32
    print(f"\nParameter reduction: {original_params:,} → {lora_params:,} trainable")
    print(f"Reduction: {100 * (1 - lora_params / original_params):.2f}%")
    
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
    
    # Configuration
    print("1. Configuration")
    print("-" * 40)
    
    config = RewardConfig()
    print(f"Hidden size: {config.hidden_size}")
    print(f"Pooling: {config.pooling}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Margin: {config.margin}")
    
    # Simple reward model
    print("\n2. Simple Reward Model")
    print("-" * 40)
    
    model = RewardModelSimple(
        vocab_size=32000,
        hidden_size=512,
        num_layers=4,
        num_heads=8,
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    # Forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 32000, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    
    rewards = model(input_ids, attention_mask)
    print(f"Reward shape: {rewards.shape}")
    print(f"Rewards: {rewards.tolist()}")
    
    # Preference loss
    print("\n3. Preference Loss")
    print("-" * 40)
    
    chosen_rewards = torch.tensor([0.5, 0.8]).to(device)
    rejected_rewards = torch.tensor([0.2, 0.3]).to(device)
    
    logits = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(logits).mean()
    
    print(f"Chosen rewards: {chosen_rewards.tolist()}")
    print(f"Rejected rewards: {rejected_rewards.tolist()}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Accuracy: {(chosen_rewards > rejected_rewards).float().mean().item():.2%}")
    
    print("\n" + "=" * 60)
    print("Reward Model Demo Complete!")
    print("=" * 60 + "\n")
    
    print("\n" + "="*60)
    print("DPO Demo")
    print("=" * 60 + "\n")
    
    # Configuration
    print("1. DPO Configuration")
    print("-" * 40)
    
    config = DPOConfig()
    print(f"Beta (KL coef): {config.beta}")
    print(f"Loss type: {config.loss_type}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Label smoothing: {config.label_smoothing}")
    
    # DPO loss computation
    print("\n2. DPO Loss Computation")
    print("-" * 40)
    
    # Simulated log probs
    policy_chosen_logps = torch.tensor([-10.0, -8.0, -12.0]).to(device)
    policy_rejected_logps = torch.tensor([-15.0, -14.0, -11.0]).to(device)
    ref_chosen_logps = torch.tensor([-11.0, -9.0, -13.0]).to(device)
    ref_rejected_logps = torch.tensor([-16.0, -15.0, -12.0]).to(device)
    
    beta = 0.1
    
    # Log ratios
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    
    print(f"Policy chosen log probs: {policy_chosen_logps.tolist()}")
    print(f"Policy rejected log probs: {policy_rejected_logps.tolist()}")
    print(f"Chosen log ratios: {chosen_logratios.tolist()}")
    print(f"Rejected log ratios: {rejected_logratios.tolist()}")
    
    # DPO logits
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()
    
    print(f"\nDPO logits: {logits.tolist()}")
    print(f"DPO loss: {loss.item():.4f}")
    
    # Implicit rewards
    chosen_rewards = beta * chosen_logratios
    rejected_rewards = beta * rejected_logratios
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    
    print(f"\nImplicit chosen rewards: {chosen_rewards.tolist()}")
    print(f"Implicit rejected rewards: {rejected_rewards.tolist()}")
    print(f"Accuracy: {accuracy.item():.2%}")
    
    # Compare loss types
    print("\n3. Loss Type Comparison")
    print("-" * 40)
    
    # Sigmoid (standard DPO)
    sigmoid_loss = -F.logsigmoid(logits).mean()
    
    # IPO
    ipo_loss = ((logits - 1 / (2 * beta)) ** 2).mean()
    
    # KTO (simplified)
    kto_loss = (1 - torch.sigmoid(logits)).mean()
    
    print(f"Sigmoid (DPO) loss: {sigmoid_loss.item():.4f}")
    print(f"IPO loss: {ipo_loss.item():.4f}")
    print(f"KTO loss: {kto_loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("DPO Demo Complete!")
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
    student_logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
    
    for loss_type in KDLossType:
        cfg = DistillationConfig(kd_loss_type=loss_type)
        loss = compute_distillation_loss(student_logits, teacher_logits, cfg)
        print(f"{loss_type.name}: {loss.item():.4f}")
    
    # Combined Loss
    print("\n3. Combined Distillation Loss")
    print("-" * 40)
    
    labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    output = combined_distillation_loss(
        student_logits, teacher_logits, labels, config
    )
    
    print(f"Total loss: {output.total_loss.item():.4f}")
    print(f"KD loss: {output.kd_loss.item():.4f}")
    print(f"CE loss: {output.ce_loss.item():.4f}")
    
    # Hidden Distillation
    print("\n4. Hidden State Distillation")
    print("-" * 40)
    
    student_hidden_size = 512
    teacher_hidden_size = 768
    
    hidden_distiller = HiddenDistillation(student_hidden_size, teacher_hidden_size).to(device)
    
    student_hidden = torch.randn(batch_size, seq_len, student_hidden_size).to(device)
    teacher_hidden = torch.randn(batch_size, seq_len, teacher_hidden_size).to(device)
    
    hidden_loss = hidden_distiller(student_hidden, teacher_hidden)
    print(f"Hidden loss: {hidden_loss.item():.4f}")
    
    # Layer Mapping
    print("\n5. Layer Mapping")
    print("-" * 40)
    
    student_layers, teacher_layers = 12, 60
    
    uniform_map = compute_layer_mapping(student_layers, teacher_layers, "uniform")
    top_heavy_map = compute_layer_mapping(student_layers, teacher_layers, "top_heavy")
    
    print("Uniform mapping (first 4):")
    for s, t in uniform_map[:4]:
        print(f"  Student {s} -> Teacher {t}")
    
    print("\nTop-heavy mapping (first 4):")
    for s, t in top_heavy_map[:4]:
        print(f"  Student {s} -> Teacher {t}")
    
    # Progressive Distillation
    print("\n6. Progressive Distillation")
    print("-" * 40)
    
    prog_config = ProgressiveConfig()
    distiller = ProgressiveDistiller(prog_config, total_steps=1000)
    
    print(f"Stages: {prog_config.num_stages}")
    print(f"Steps per stage: {distiller.steps_per_stage}")
    
    for step in [0, 333, 666, 999]:
        while distiller.current_step < step:
            distiller.step()
        state = distiller.get_state()
        print(f"Step {step}: stage={state['stage']}, temp={state['temperature']:.2f}")
    
    # Temperature Schedule
    print("\n7. Temperature Schedule")
    print("-" * 40)
    
    schedules = [
        ("Constant", TemperatureScheduler(4.0, 4.0, 1000, "constant")),
        ("Linear", TemperatureScheduler(6.0, 2.0, 1000, "linear")),
        ("Cosine", TemperatureScheduler(6.0, 2.0, 1000, "cosine")),
    ]
    
    for name, schedule in schedules:
        print(f"{name}:")
        for step in [0, 250, 500, 750, 1000]:
            print(f"  Step {step}: T={schedule.get_temperature(step):.2f}")
    
    print("\n" + "=" * 60)
    print("Distillation Demo Complete!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_demos()
