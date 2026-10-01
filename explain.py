import os
import torch
import numpy as np
from PIL import Image

from torchvision import transforms
from torchvision.models import resnet18

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_PATH = r"models\resnet18_authenticity.pth"
OUTPUT_DIR = r"outputs"

IMAGE_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# ============================================================
# 3. LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

class_names = checkpoint["class_names"]

model = resnet18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# ============================================================
# 4. IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 5. PREDICTION + GRAD-CAM
# ============================================================

def explain_image(image_path):

    print("\nProcessing image...")
    print(f"Input image: {image_path}")

    original_image = Image.open(
        image_path
    ).convert("RGB")

    resized_image = original_image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    rgb_image = np.array(
        resized_image
    ).astype(np.float32) / 255.0

    input_tensor = transform(
        original_image
    ).unsqueeze(0).to(device)


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()


    predicted_label = class_names[
        predicted_class
    ]

    print("\nPrediction:")
    print(f"Class      : {predicted_label}")
    print(
        f"Confidence : {confidence * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    target_layers = [
        model.layer4[-1]
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )


    # --------------------------------------------------------
    # Save heatmap
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        "gradcam_result.jpg"
    )

    Image.fromarray(
        visualization
    ).save(output_path)

    print("\nExplainability result:")
    print(f"Grad-CAM saved to:")
    print(output_path)

    return (
        predicted_label,
        confidence,
        output_path
    )


# ============================================================
# 6. COMMAND-LINE USAGE
# ============================================================

if __name__ == "__main__":

    image_path = input(
        "\nEnter image path: "
    ).strip().strip('"')

    if not os.path.exists(image_path):

        print(
            "\nERROR: Image file not found."
        )

    else:

        explain_image(
            image_path
        )