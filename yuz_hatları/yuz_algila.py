#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt


def draw_landmarks_on_image(rgb_image, detection_result, etiket):
  # bulunna yüzler ve o yüzler üzerindeki koordinatlar
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)
  # print("Bulunan yüz sayısı", len(face_landmarks_list))
  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]
    # print("Nokta sayısı", len(face_landmarks))

    # Sadece x,y ve z koordinatlarını al
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    # print(len([landmark.x for landmark in face_landmarks]))
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    koordinatlar= []
    for landmark in face_landmarks:
       koordinatlar.append(str(round(landmark.x, 4)))
       koordinatlar.append(str(round(landmark.y,4)))
    
    koordinatlar = ",".join(koordinatlar)
    koordinatlar += f",{etiket}\n"
    with open("veriseti.csv", "a") as f:
       f.write(koordinatlar)

    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp.solutions.drawing_styles
    #     .get_default_face_mesh_tesselation_style())


    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp.solutions.drawing_styles
    #     .get_default_face_mesh_contours_style())
    

    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_IRISES,
    #       landmark_drawing_spec=None,
    #       connection_drawing_spec=mp.solutions.drawing_styles
    #       .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()


def sutun_basliklarini_olustur():
  with open("veriseti.csv", "w") as f:
      satir = ""
      for i in range(1, 479):
        satir = satir + f"x{i},y{i},"
      satir = satir + "Etiket\n"
      f.write(satir)


ifadeler = ["happy", "sad", "angry", "surprised"]
ilk_ifade = True



# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

for etiket in ifadeler:
        if ilk_ifade:
            sutun_basliklarini_olustur()
            ilk_ifade = False

        print(f"\n\u25B6\ufe0f Simdi '{etiket.upper()}' ifadesi toplanacak. Çıkmak için 'q' tuşuna basın.")
        print("kameradan okuma yapmaya çalıştıkkkkkkk")
        cam = cv2.VideoCapture(0)
        sayac = 0
        print("kameradan okuma yapmaya çalıştıkkkkkkk")
# Kameradan görüntü alımı
        while cam.isOpened() and sayac < 200:
            print("kameradan okuma yapmaya çalıştıkkkkk")
            basari, frame = cam.read()
            print("kameradan okuma yapmaya çalıştıkkkkk")
            if not basari:
                print("kameradan okuma yaptı")
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                # STEP 4: Detect face landmarks from the input image.
            detection_result = detector.detect(mp_image)

                # STEP 5: Process the detection result. In this case, visualize it.
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result,etiket)
            frame_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, f"Toplanan ifade: {etiket}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Ornek sayisi: {sayac}/200", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Yuz", frame_bgr)

            if detection_result.face_landmarks:
                sayac += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                

        cam.release()
        cv2.destroyAllWindows() 
        