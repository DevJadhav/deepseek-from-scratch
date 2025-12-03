import torch
import coremltools as ct
from deepseek.model.attention import MultiQueryAttention

def export_mqa():
    print("Exporting MQA to CoreML...")
    d_model = 512
    num_heads = 8
    model = MultiQueryAttention(d_model, num_heads)
    model.eval()
    
    # CoreML usually expects fixed shapes or flexible shapes within ranges
    # For this demo, we use a fixed shape
    example_input = torch.randn(1, 64, 512)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    print("Converting...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="x", shape=example_input.shape)],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL, # Use Neural Engine if possible
    )
    
    mlmodel.save("MQA.mlpackage")
    print("Saved MQA.mlpackage")

if __name__ == "__main__":
    try:
        export_mqa()
    except Exception as e:
        print(f"Export failed (likely due to missing coremltools or environment issues): {e}")
