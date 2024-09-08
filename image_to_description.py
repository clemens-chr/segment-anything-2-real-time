import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import requests
import tyro
from PIL import Image


@dataclass
class ImageToDescriptionArgs:
    image_path: Path = (
        Path(__file__).parent / "outputs" / "rendered_mesh_with_white_background.png"
    )
    model: Literal["gpt-4o-mini", "gpt-40"] = "gpt-4o-mini"
    max_tokens: int = 300
    openai_api_key: Optional[str] = None

    def __post_init__(self) -> None:
        assert self.image_path.exists(), f"{self.image_path} does not exist"


def get_openai_api_key() -> str:
    """
    Retrieve the API key for the OpenAI client. It first tries to get it from the environment variable.
    If not found, it attempts to read it from a local file.

    Returns:
        str: The API key as a string.

    Raises:
        FileNotFoundError: If the API key file is not found.
        AssertionError: If no API key is found.
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key is None:
        print("Did not find OPENAI_API_KEY")

        API_KEY_FILE = "openai_api_key.txt"
        api_key_filepath = Path(__file__).parent / API_KEY_FILE
        if not api_key_filepath.exists():
            raise FileNotFoundError(f"API key file not found at {api_key_filepath}")
        with open(api_key_filepath, "r") as f:
            openai_api_key = f.read().strip()

    assert (
        openai_api_key is not None
    ), "Please set the OPENAI_API_KEY environment variable"
    return openai_api_key


def encode_image(image: Image.Image) -> str:
    """
    Encodes a PIL image to a base64 string.

    Args:
        image (PIL.Image.Image): The image to encode.

    Returns:
        str: The base64-encoded image.
    """
    with io.BytesIO() as image_buffer:
        image.save(image_buffer, format="PNG")
        return base64.b64encode(image_buffer.getvalue()).decode("utf-8")


def generate_image_description(
    image: Image.Image, openai_api_key: Optional[str], model: str, max_tokens: int
) -> Tuple[str, dict]:
    """
    Generate a description for the provided image using the OpenAI API.

    Args:
        image (PIL.Image.Image): The image for which to generate a description.
        openai_api_key (Optional[str]): The API key for the OpenAI API, if not provided, it will be read from the environment.
        model (str): The OpenAI model to use.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        Tuple[str, dict]: The description and the full response from the OpenAI API.

    Raises:
        Exception: If the request fails.
    """
    if openai_api_key is None:
        openai_api_key = get_openai_api_key()

    # Encode the image to base64
    base64_image = encode_image(image)

    # Set up headers and payload for the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the primary object on the table (in as few words as possible)? E.g. Purple block, blue pitcher, yellow marker, etc.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
    }

    # Send the request to OpenAI API
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    if response.status_code == 200:
        response_json = response.json()
        description = response_json["choices"][0]["message"]["content"]
        return description, response_json
    else:
        raise Exception(
            f"Request failed with status code {response.status_code}: {response.text}"
        )


def main() -> None:
    """
    Main function to retrieve an image description.
    """
    # Parse command-line arguments
    args = tyro.cli(ImageToDescriptionArgs)
    print("=" * 80)
    print(f"args:\n{tyro.extras.to_yaml(args)}")
    print("=" * 80 + "\n")

    # Open the image using PIL
    with Image.open(args.image_path) as image:
        try:
            description, response = generate_image_description(
                image=image,
                openai_api_key=args.openai_api_key,
                model=args.model,
                max_tokens=args.max_tokens,
            )
            print("Response from OpenAI API:", response)
            print(f"Description: {description}")
        except Exception as e:
            print("An error occurred:", e)


if __name__ == "__main__":
    main()
