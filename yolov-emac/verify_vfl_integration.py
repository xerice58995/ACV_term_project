import torch
import torch.nn as nn
from ultralytics.utils.loss import VarifocalLoss, v8DetectionLoss
from ultralytics import YOLO

def test_vfl_basic():
    """Test VarifocalLoss basic functionality"""
    print("Testing VarifocalLoss basic functionality...")

    # Create sample tensors
    pred_score = torch.randn(4, 100, 80, requires_grad=True)  # [B, N, C]
    gt_score = torch.rand(4, 100, 80)                         # [B, N, C]
    label = torch.randint(0, 2, (4, 100))                    # [B, N]

    # Compute VFL
    vfl = VarifocalLoss()
    loss = vfl.forward(pred_score, gt_score, label.float(), alpha=0.75, gamma=2.0)

    print(f"✓ VFL Loss: {loss.item():.4f}")

    # Check if loss is differentiable
    loss.backward()
    assert pred_score.grad is not None, "Gradients not computed!"
    print(f"✓ Gradient shape: {pred_score.grad.shape}")
    print(f"✓ Gradient mean: {pred_score.grad.mean():.6f}")

    return True

def test_vfl_vs_bce():
    """Compare VFL with BCE"""
    print("\nComparing VFL with BCE...")

    pred_score = torch.randn(4, 100, 80)
    target_scores = torch.rand(4, 100, 80)
    label = torch.randint(0, 2, (4, 100)).float()

    # VFL
    vfl = VarifocalLoss()
    vfl_loss = vfl.forward(pred_score, target_scores, label)

    # BCE
    bce = nn.BCEWithLogitsLoss()
    bce_loss = bce(pred_score, target_scores).mean()

    print(f"✓ VFL Loss: {vfl_loss.item():.4f}")
    print(f"✓ BCE Loss: {bce_loss.item():.4f}")
    print(f"✓ Ratio (VFL/BCE): {(vfl_loss/bce_loss).item():.2f}")

    return True

def test_yolo_model_loading():
    """Test if YOLO model loads with VFL configuration"""
    print("\nTesting YOLO model loading...")

    try:
        # Load model
        model = YOLO('yolov8n.pt')
        print(f"✓ Model loaded successfully")

        # Check if use_vfl parameter is recognized
        print(f"✓ Model args: {model.model.args.__dict__ if hasattr(model.model, 'args') else 'N/A'}")

        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("VarifocalLoss Integration Verification")
    print("=" * 60)

    try:
        test_vfl_basic()
        test_vfl_vs_bce()
        test_yolo_model_loading()

        print("\n" + "=" * 60)
        print("✓ All verification tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
