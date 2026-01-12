# Chapter 13: Knowledge Distillation

## Overview

Knowledge distillation enables training smaller, efficient models that retain most of the capabilities of larger teacher models. This chapter covers distillation techniques used in DeepSeek model compression.

**Key Papers:**
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton et al., 2015)
- [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108) (Sanh et al., 2019)
- [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351) (Jiao et al., 2019)
- [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525) (Gou et al., 2020)

## Table of Contents

1. [Foundations of Knowledge Distillation](#foundations)
2. [Standard Knowledge Distillation](#standard-kd)
3. [Sequence-Level Distillation](#seq-kd)
4. [Feature Distillation](#feature-kd)
5. [Progressive Distillation](#progressive)
6. [Implementation Guide](#implementation)
7. [Best Practices](#best-practices)

---

<a name="foundations"></a>
## 1. Foundations of Knowledge Distillation

### 1.1 Core Concept

Knowledge distillation transfers knowledge from a large **teacher** model to a smaller **student** model:

```
Teacher Model (Large)
       │
       ▼
   Soft Labels / Features
       │
       ▼
Student Model (Small)
```

### 1.2 Why Distillation?

| Aspect | Teacher | Student |
|--------|---------|---------|
| Parameters | 67B+ | 1.5B-7B |
| Latency | High | Low |
| Memory | Large | Small |
| Quality | Best | Near-teacher |

### 1.3 Types of Knowledge Transfer

```
┌─────────────────────────────────────────────────────────┐
│                Knowledge Types                          │
├─────────────────────────────────────────────────────────┤
│  Output Level   │ Soft labels, logits, probabilities    │
├─────────────────┼───────────────────────────────────────┤
│  Feature Level  │ Hidden states, activations            │
├─────────────────┼───────────────────────────────────────┤
│  Attention      │ Attention patterns, head outputs      │
├─────────────────┼───────────────────────────────────────┤
│  Relation       │ Token relationships, representations  │
└─────────────────┴───────────────────────────────────────┘
```

---

<a name="standard-kd"></a>
## 2. Standard Knowledge Distillation

### 2.1 Soft Label Distillation

The original formulation by Hinton et al. uses **temperature-scaled softmax**:

$$p_i^T = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

Where:
- $z_i$ = logits
- $T$ = temperature (higher → softer distribution)

### 2.2 KL Divergence Loss

The distillation loss measures divergence between teacher and student distributions:

$$\mathcal{L}_{KD} = T^2 \cdot \text{KL}\left(p^T_{teacher} \| p^T_{student}\right)$$

$$= T^2 \sum_i p^T_{teacher,i} \log\frac{p^T_{teacher,i}}{p^T_{student,i}}$$

The $T^2$ factor compensates for gradient magnitude changes with temperature.

### 2.3 Combined Loss

Combine distillation with standard cross-entropy:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{KD} + (1 - \alpha) \cdot \mathcal{L}_{CE}$$

**Typical values:**
- $\alpha \in [0.5, 0.9]$ (higher for larger students)
- $T \in [2, 10]$ (higher for more smoothing)

### 2.4 Alternative Distance Metrics

**Jensen-Shannon Divergence (symmetric):**

$$\text{JSD}(P \| Q) = \frac{1}{2}\text{KL}(P \| M) + \frac{1}{2}\text{KL}(Q \| M)$$

where $M = \frac{P + Q}{2}$

**Mean Squared Error:**

$$\mathcal{L}_{MSE} = \|p^T_{teacher} - p^T_{student}\|^2$$

**Cosine Similarity:**

$$\mathcal{L}_{cos} = 1 - \frac{z_{teacher} \cdot z_{student}}{\|z_{teacher}\| \|z_{student}\|}$$

### 2.5 Implementation

```python
def kd_loss(student_logits, teacher_logits, temperature=4.0):
    """Standard knowledge distillation loss."""
    # Scale by temperature
    student_soft = F.softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL divergence
    kl = F.kl_div(
        student_soft.log(),
        teacher_soft,
        reduction='batchmean'
    )
    
    # Scale by T²
    return kl * (temperature ** 2)
```

---

<a name="seq-kd"></a>
## 3. Sequence-Level Knowledge Distillation

### 3.1 Motivation

Token-level distillation has limitations:
- Requires aligned sequences
- Teacher may assign mass to unlikely tokens
- Doesn't capture generation quality

### 3.2 SeqKD Approach

Use teacher-generated sequences as training data:

```
┌──────────────────────────────────────────────────────┐
│  Prompt  →  Teacher Model  →  Generated Response    │
│                                     │               │
│                                     ▼               │
│              Student trains on (Prompt, Response)   │
└──────────────────────────────────────────────────────┘
```

### 3.3 SeqKD Loss

Standard cross-entropy on teacher generations:

$$\mathcal{L}_{SeqKD} = -\sum_t \log p_{student}(y_t^{teacher} | y_{<t}^{teacher}, x)$$

### 3.4 Sampling Strategies

**Beam Search:**
- Deterministic
- High quality
- Less diverse

**Nucleus Sampling:**
- Stochastic
- More diverse
- May include errors

**Multiple Samples:**
- Generate N responses
- Filter by quality
- Train on best

### 3.5 Mixed Training

Combine teacher generations with ground truth:

$$\mathcal{L}_{mixed} = \beta \cdot \mathcal{L}_{SeqKD} + (1-\beta) \cdot \mathcal{L}_{GT}$$

Typical $\beta \in [0.3, 0.7]$

---

<a name="feature-kd"></a>
## 4. Feature Distillation

### 4.1 Hidden State Distillation

Match intermediate representations:

$$\mathcal{L}_{hidden} = \sum_{l \in \mathcal{L}} \|\mathbf{W}_l h_l^{student} - h_{\phi(l)}^{teacher}\|^2$$

Where:
- $\mathcal{L}$ = student layers to distill
- $\phi(l)$ = maps student layer to teacher layer
- $\mathbf{W}_l$ = projection matrix (if dimensions differ)

### 4.2 Layer Mapping Strategies

**Uniform Mapping:**
```
Student  Teacher (60 layers)
Layer 0  →  Layer 0
Layer 1  →  Layer 5
Layer 2  →  Layer 10
...
Layer 11 →  Layer 55
```

**Top-Heavy Mapping:**
```
Student  Teacher
Layer 0  →  Layer 0
Layer 1  →  Layer 2
...
Layer 6  →  Layer 20
Layer 7  →  Layer 30
...
Layer 11 →  Layer 59
```

### 4.3 Attention Distillation

Match attention patterns:

$$\mathcal{L}_{attn} = \sum_{l,h} \text{KL}(A_{l,h}^{teacher} \| A_{l,h}^{student})$$

For models with different head counts, average over heads:

$$\bar{A}_l = \frac{1}{H} \sum_{h=1}^{H} A_{l,h}$$

### 4.4 Implementation

```python
def hidden_distillation_loss(
    student_hidden,    # (batch, seq, d_student)
    teacher_hidden,    # (batch, seq, d_teacher)
    projection=None    # (d_student, d_teacher)
):
    """Distill hidden state representations."""
    if projection is not None:
        student_proj = student_hidden @ projection
    else:
        student_proj = student_hidden
    
    return F.mse_loss(student_proj, teacher_hidden)
```

---

<a name="progressive"></a>
## 5. Progressive Distillation

### 5.1 Concept

Train a sequence of increasingly smaller models:

```
Teacher (67B)
    ↓
Intermediate (32B)
    ↓
Intermediate (14B)
    ↓
Student (7B)
```

### 5.2 Benefits

1. **Easier optimization**: Smaller capacity gaps
2. **Better feature transfer**: Intermediate representations more similar
3. **Curriculum effect**: Gradually harder compression

### 5.3 Temperature Scheduling

Decrease temperature as training progresses:

**Linear Schedule:**
$$T(t) = T_{start} + (T_{end} - T_{start}) \cdot \frac{t}{t_{max}}$$

**Cosine Schedule:**
$$T(t) = T_{end} + \frac{T_{start} - T_{end}}{2}\left(1 + \cos\frac{\pi t}{t_{max}}\right)$$

### 5.4 Stage Configuration

```python
stages = [
    {"intermediate_size": 7168, "teacher": "67B", "temp": 6.0},
    {"intermediate_size": 4096, "teacher": "32B", "temp": 4.0},
    {"intermediate_size": 2048, "teacher": "14B", "temp": 2.0},
]
```

### 5.5 When to Advance Stages

- Fixed steps per stage
- Validation loss plateau
- Student-teacher gap threshold

---

<a name="implementation"></a>
## 6. Implementation Guide

### 6.1 Complete Distillation Pipeline

```python
class DistillationTrainer:
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        config: DistillationConfig,
    ):
        self.student = student
        self.teacher = teacher
        self.config = config
        
        # Freeze teacher
        for param in teacher.parameters():
            param.requires_grad = False
    
    def compute_loss(
        self,
        input_ids: Tensor,
        labels: Tensor,
    ) -> Dict[str, Tensor]:
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids,
                output_hidden_states=self.config.hidden_distill,
                output_attentions=self.config.attention_distill,
            )
        
        # Student forward
        student_outputs = self.student(
            input_ids,
            output_hidden_states=self.config.hidden_distill,
            output_attentions=self.config.attention_distill,
        )
        
        losses = {}
        
        # Output distillation
        losses['kd'] = self.kd_loss(
            student_outputs.logits,
            teacher_outputs.logits,
        )
        
        # Hard label loss
        if self.config.use_hard_labels:
            losses['ce'] = F.cross_entropy(
                student_outputs.logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        
        # Hidden distillation
        if self.config.hidden_distill:
            losses['hidden'] = self.hidden_loss(
                student_outputs.hidden_states,
                teacher_outputs.hidden_states,
            )
        
        # Combined loss
        total = (
            self.config.alpha * losses['kd'] +
            (1 - self.config.alpha) * losses.get('ce', 0) +
            self.config.hidden_weight * losses.get('hidden', 0)
        )
        
        return {'total': total, **losses}
```

### 6.2 Memory-Efficient Distillation

For large teachers, use gradient checkpointing and mixed precision:

```python
def distillation_step(batch):
    # Teacher in eval mode, FP16
    with torch.no_grad(), torch.cuda.amp.autocast():
        teacher_logits = teacher(batch['input_ids']).logits
    
    # Student with gradient checkpointing
    with torch.cuda.amp.autocast():
        student_logits = student(
            batch['input_ids'],
            use_cache=False,
        ).logits
        
        loss = kd_loss(student_logits, teacher_logits)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 6.3 Distributed Distillation

When teacher doesn't fit on single GPU:

```python
# Option 1: Model parallelism for teacher
teacher = load_model_parallel(teacher_path)

# Option 2: Precompute teacher logits
def precompute_teacher_logits(dataset, teacher, save_path):
    for batch in dataset:
        with torch.no_grad():
            logits = teacher(batch['input_ids']).logits
        torch.save(logits, save_path / f"{batch_idx}.pt")

# Option 3: Online distillation with smaller teacher batches
def online_distillation(batch, teacher, student):
    teacher_batch_size = batch_size // 4
    all_teacher_logits = []
    
    for i in range(0, batch_size, teacher_batch_size):
        mini_batch = batch[i:i+teacher_batch_size]
        with torch.no_grad():
            logits = teacher(mini_batch).logits
        all_teacher_logits.append(logits)
    
    teacher_logits = torch.cat(all_teacher_logits)
    # Continue with student training...
```

---

<a name="best-practices"></a>
## 7. Best Practices

### 7.1 Hyperparameter Selection

| Parameter | Range | Notes |
|-----------|-------|-------|
| Temperature | 2-10 | Higher for more smoothing |
| Alpha | 0.5-0.9 | Higher when student is larger |
| Learning Rate | 1e-5 - 5e-5 | Lower than pretraining |
| Hidden Weight | 0.1-0.5 | If using hidden distillation |

### 7.2 Training Curriculum

1. **Start with output distillation only**
2. **Add hidden distillation** after convergence
3. **Fine-tune with lower temperature**
4. **Final fine-tune on downstream tasks**

### 7.3 Quality Metrics

Monitor during training:
- **Teacher-student KL divergence**
- **Token accuracy vs teacher**
- **Generation quality (BLEU, ROUGE)**
- **Downstream task performance**

### 7.4 Common Pitfalls

| Issue | Solution |
|-------|----------|
| Student doesn't learn | Lower temperature, higher alpha |
| Capacity gap too large | Use progressive distillation |
| Hidden mismatch | Better layer mapping |
| Overfitting to teacher | Add data augmentation |

### 7.5 DeepSeek Distillation Recipe

Based on DeepSeek papers:

1. **Data**: Mix of pretraining and instruction data
2. **Temperature**: Start at 6, decay to 2
3. **Alpha**: 0.7 for output, 0.3 for CE
4. **Hidden**: Distill every 4th layer
5. **Stages**: 3-stage progressive (67B → 32B → 7B)

---

## Summary

Knowledge distillation enables:

1. **Model compression** without significant quality loss
2. **Faster inference** with smaller models
3. **Knowledge transfer** from specialized to general models

Key techniques:

| Technique | Best For |
|-----------|----------|
| Standard KD | General compression |
| SeqKD | Generation tasks |
| Feature KD | Representation learning |
| Progressive | Large compression ratios |

## Next Steps

- Implement distillation pipeline
- Experiment with temperature schedules
- Try progressive distillation for extreme compression
- Combine with quantization for maximum efficiency
