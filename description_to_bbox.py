from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import cv2
import numpy as np
import sam2
import supervision as sv
import torch
import tyro
from dds_cloudapi_sdk import (
    Client,
    Config,
    DetectionModel,
    DetectionTarget,
    DetectionTask,
    TextPrompt,
)
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


@dataclass
class GroundedSAM2DemoConfig:
    grounding_model: Literal["gdino", "gdino1.5"] = "gdino"
    text_prompt: str = "block."
    image_path: Path = Path(__file__).parent / "example_data" / "woodblock.png"
    sam2_checkpoint: Path = (
        Path(sam2.__file__).parents[1] / "checkpoints/sam2_hiera_large.pt"
    )
    sam2_model_config: str = "sam2_hiera_l.yaml"
    output_dir: Path = Path("outputs/grounded_sam2_demo")
    gdino_1_5_api_token: Optional[str] = None

    def __post_init__(self) -> None:
        assert self.image_path.exists(), f"{self.image_path} does not exist"
        assert self.sam2_checkpoint.exists(), f"{self.sam2_checkpoint} does not exist"
        self.output_dir.mkdir(parents=True, exist_ok=True)


def generate_bbox(
    image: Image.Image,
    text_prompt: str,
    grounding_model: Literal["gdino", "gdino1.5"],
    gdino_1_5_api_token: Optional[str],
) -> Tuple[np.ndarray, list, list]:
    # Environment setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Needs to have a . in text prompt
    if "." not in text_prompt:
        text_prompt += "."
        print(f"Added period to text prompt: {text_prompt} since it's required")

    # Step 3: Grounding DINO or Grounding DINO 1.5
    if grounding_model == "gdino":
        processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        ).to(device)

        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]],
        )

        bboxes = results[0]["boxes"].cpu().numpy()
        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]

        # Keep best only
        # Find the index of the maximum score
        max_confidence_index = np.argmax(confidences).item()
        bboxes = bboxes[max_confidence_index : max_confidence_index + 1]
        confidences = confidences[max_confidence_index : max_confidence_index + 1]
        class_names = class_names[max_confidence_index : max_confidence_index + 1]

    elif grounding_model == "gdino1.5":
        assert (
            gdino_1_5_api_token is not None
        ), "Please provide an API token to use Grounding DINO 1.5"

        client = Client(Config(gdino_1_5_api_token))

        # Must save to file to upload
        image_path = Path("/tmp/image.png")
        image.save(image_path)

        image_url = client.upload_file(str(image_path))
        task = DetectionTask(
            image_url=image_url,
            prompts=[TextPrompt(text=text_prompt)],
            targets=[DetectionTarget.BBox],
            model=DetectionModel.GDino1_5_Pro,
        )
        client.run_task(task)
        objects = task.result.objects

        bboxes = np.array([obj.bbox for obj in objects])
        confidences = [obj.score for obj in objects]
        class_names = [obj.category for obj in objects]
    else:
        raise ValueError(f"Invalid grounding model: {grounding_model}")

    assert bboxes.shape == (1, 4), f"{bboxes.shape}"
    assert len(confidences) == 1, f"{len(confidences)}"
    assert len(class_names) == 1, f"{len(class_names)}"

    return bboxes, confidences, class_names


def generate_mask(
    image: Image.Image,
    sam2_model_config: str,
    sam2_checkpoint: Path,
    bboxes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Step 1: Initialize SAM2
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    sam2_predictor.set_image(np.array(image.convert("RGB")))

    # Step 4: Use SAM2 to predict masks
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None, point_labels=None, box=bboxes, multimask_output=False
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    width, height = image.size
    assert masks.shape == (1, height, width), f"{masks.shape}"
    return masks, scores, logits


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def visualize_results(
    image: Image.Image,
    bboxes: np.ndarray,
    confidences: list,
    class_names: list,
    masks: np.ndarray,
    output_dir: Path,
) -> None:
    cv2_image = pil_to_cv2(image)

    # Visualize
    class_ids = np.array(list(range(len(class_names))))
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(class_names, confidences)
    ]

    # Step 5: Visualize the results
    detections = sv.Detections(xyxy=bboxes, mask=masks.astype(bool), class_id=class_ids)

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(
        scene=cv2_image.copy(), detections=detections
    )

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )

    output_image_path = output_dir / "annotated_image.jpg"
    cv2.imwrite(
        str(output_image_path),
        annotated_frame,
    )

    output_image_with_mask_path = output_dir / "annotated_image_with_mask.jpg"
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    cv2.imwrite(
        str(output_image_with_mask_path),
        annotated_frame,
    )

    print(
        f"Saved images to {output_dir}: {output_image_path}, {output_image_with_mask_path}"
    )


def main() -> None:
    # Parse command-line arguments
    config = tyro.cli(GroundedSAM2DemoConfig)
    print("=" * 80)
    print(f"Parsed Arguments:\n{tyro.extras.to_yaml(config)}")
    print("=" * 80 + "\n")

    image = Image.open(config.image_path)

    # Run the demo with the parsed config
    bboxes, confidences, class_names = generate_bbox(
        image=image,
        text_prompt=config.text_prompt,
        grounding_model=config.grounding_model,
        gdino_1_5_api_token=config.gdino_1_5_api_token,
    )
    masks, _, _ = generate_mask(
        image=image,
        sam2_model_config=config.sam2_model_config,
        sam2_checkpoint=config.sam2_checkpoint,
        bboxes=bboxes,
    )
    visualize_results(
        image=image,
        bboxes=bboxes,
        confidences=confidences,
        class_names=class_names,
        masks=masks,
        output_dir=config.output_dir,
    )


if __name__ == "__main__":
    main()
