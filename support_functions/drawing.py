
import cv2
import os
def draw_bounding_box_on_first_image(df,image_folder):
  """Draws a bounding box on the first image from a Pandas dataset.

  Args:
    df: A Pandas DataFrame containing the image name, bounding box coordinates, and label for each image.
    df has to have following format  image_path ,x1,y1,x2,y2,label
  Returns:
    None.
  """

  # Get the image name, bounding box coordinates, and label for the first image.
  img_name = df.iloc[0]['img_name']
  x1 = df.iloc[0]['x1']
  y1 = df.iloc[0]['y1']
  x2 = df.iloc[0]['x2']
  y2 = df.iloc[0]['y2']
  label = df.iloc[0]['label']
  path = os.path.join(image_folder,img_name)

  # Read the image.
  img = cv2.imread(path)

  # Draw the bounding box.
  cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

  # Write the label above the bounding box.
  cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  # Display the image.
  cv2.imshow('Image with bounding box', img)
  cv2.waitKey(0)

