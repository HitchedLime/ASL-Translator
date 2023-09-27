import torch

def collate_fn(batch):
  """Collate function that returns a tensor of the 4 images.

  Args:
    batch: A list of tuples, where each tuple contains two elements:
      - A PyTorch tensor of the image data.
      - A dictionary of the bounding boxes and labels for the image.

  Returns:
    A dictionary containing the following keys:
      - images: A PyTorch tensor of the 4 images, stacked together.
      - targets: A list of dictionaries, where each dictionary contains the
        bounding boxes and labels for a single object in the image.
  """

  images = []
  targets = []

  for image, target in batch:
    # Normalize the image to the range [0, 1]
    image = image / 255.0

    images.append(image)
    targets.append(target)

  # Stack the images together
  images = torch.stack(images)

  return {
      "images": images,
      "targets": targets,
  }
