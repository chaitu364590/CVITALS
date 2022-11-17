import cv2
import streamlit as st
import tensorflow as tf
import os
import numpy as np
import PIL
import pandas as pd
from PIL import Image,ImageEnhance
## Page Title
#st.set_page_config(page_title = "Cats vs Dogs Image Classification")
st.title("CONTACTLESS VITAL SIGN DETECTION")
st.markdown("------")

model_path='vitals.tflite'



# Load the labels into a list
classes = ['watch glass', 'conical flask','cork','bunsen burners','corks','testtube holder','cap','caps','chemicals','beaker','testtubes','burutte','gloves','burette','testtube','bunsen burner','flame','glove','pippete','testtube stand']
#label_map = model.model_spec.config.label_map
#for label_id, label_name in label_map.as_dict().items():
 # classes[label_id-1] = label_name
uploaded_video = st.file_uploader("Choose video", type=["mp4", "mov"])
frame_skip = 300 # display every 300 frames

if uploaded_video is not None: # run only when user uploads video
    vid = uploaded_video.name
    with open(vid, mode='wb') as f:
        f.write(uploaded_video.read()) # save video to disk

    st.markdown(f"""
    ### Files
    - {vid}
    """,
    unsafe_allow_html=True) # display file name

    vidcap = cv2.VideoCapture(vid) # load video from disk
    cur_frame = 0
    success = True
    

    while success:
        success, frame = vidcap.read() # get next frame from video
        if cur_frame % frame_skip == 0: # only analyze every n=300 frames
            print('frame: {}'.format(cur_frame)) 
            pil_img = Image.fromarray(frame) # convert opencv frame (with type()==numpy) i
            st.image(pil_img)
            cur_frame += 1
            our_image_A = pil_img##tview
            st.image(our_image_A)
            our_image_A=np.array(our_image_A.convert('RGB'))
            gray = cv2.cvtColor(our_image_A, cv2.COLOR_BGR2GRAY)
            st.image(gray)

            gray8_image=np.zeros((120, 160), dtype=np.uint8)
            gray8_image=cv2.normalize(gray, gray8_image,0,255,cv2.NORM_MINMAX)
            gray8_image=np.uint8(gray8_image)
            inferno_palette=cv2.applyColorMap(gray8_image, cv2.COLORMAP_INFERNO)
            jet_palette=cv2.applyColorMap(gray8_image, cv2.COLORMAP_JET)
            viridis_palette=cv2.applyColorMap(gray8_image, cv2.COLORMAP_VIRIDIS)
            st.image(gray8_image)
            st.image(inferno_palette)
            st.image(jet_palette)
            st.image(viridis_palette)
# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

def preprocess_image(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  return resized_img, original_image


def set_input_tensor(interpreter, image):
  """Set the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Retur the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  # Feed the input image to the model
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all outputs from the model
  scores = get_output_tensor(interpreter, 0)
  boxes = get_output_tensor(interpreter, 1)
  count = int(get_output_tensor(interpreter, 2))
  classes = get_output_tensor(interpreter, 3)

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
      
  return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.3):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path, 
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)
  
  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute 
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])

    # Find the class index of the current object
    class_id = int(obj['class_id'])
    st.write(classes[class_id])

    # Draw the bounding box and label on the image
    color = [int(c) for c in COLORS[class_id]]
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
    cv2.putText(original_image_np, label, (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8


## Input Fields
uploaded_file = st.file_uploader("Upload a Image", type=["jpg","png", 'jpeg'])
if uploaded_file is not None:
    with open(os.path.join("/tmp",uploaded_file.name),"wb") as f:
        f.write(uploaded_file.getbuffer())
    path = os.path.join("tmp",uploaded_file.name)
    URL =path
    DETECTION_THRESHOLD = 0.3

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Run inference and draw detection result on the local copy of the original file
    detection_result_image = run_odt_and_draw_results(
        URL, 
        interpreter, 
        threshold=DETECTION_THRESHOLD
    )

    # Show the detection result
    st.image(detection_result_image)

 













