from hand_pose_tools import *
from threading import Thread
from multiprocessing import Process

def realTimeCalibration(camera_selection, SIZE):
    global STOP_ALL_PROCESSES, global_image, K, extended_K, dist, calib_R_t_with_0001

    while not STOP_ALL_PROCESSES:
        global_image, K, extended_K, calib_R_t_with_0001, dist = ImageHandler.getCalibrationMatrices(camera_selection, SIZE)

def realTimeEstimation(camera_selection, mode_selection):
    global STOP_ALL_PROCESSES, global_image, K, extended_K, dist, calib_R_t_with_0001
    
    if mode_selection == 1: # 멀티 스레드
        with mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands: # 손을 추적할 모델입니다.
            while True:
                results, image, h, w, _ = ImageHandler.processImage(hands, global_image) # 랜드마크 추적결과, 이미지, 이미지세로, 이미지가로
                if results.multi_hand_landmarks:  # 손(들)을 찾았다면
                    for i, hand_landmarks in enumerate(results.multi_hand_landmarks): # 손(들)을 돌면서
                        # 한손이면 i는 0, 양손이면 i는 0, 1
                        # hand_landmarks는 [노멀라이즈드랜드마크, 노멀라이즈드랜드마크, 노멀라이즈드랜드마크, ... ] 처럼
                        # NormalizedLandmark 여러개를 담은 리스트입니다.(hand_landmark's' 이기떄문에 복수형)
                        # Python의 List<NormalizedLandmarkList>
                        
                        model_points = np.float32([[l.x, l.y, l.z] for l in results.multi_hand_world_landmarks[i].landmark])  # Hand 3D 좌표
                        image_points = np.float32([[l.x * w, l.y * h] for l in hand_landmarks.landmark])  # Hand 2D 좌표

                        # H2C에서 H곱해서 나온 C좌표를 2D이미지로 찍을때 어떤 픽셀에 찍어야 할지 리스트와
                        # C2W에서 C곱해서 나온 W좌표를 2D이미지로 찍을때 어떤 픽셀에 찍어야 할지 리스트
                        # 두개를 리턴받습니다.
                        h2c2i_coords, w2c2i_coords = MatrixHandler.getCoordinates(K, extended_K, dist, calib_R_t_with_0001, model_points, image_points)

                        # 시각적 분석
                        image = Analyzer.visualAnalyze(image_points, h2c2i_coords, w2c2i_coords, h, w, image)

                cv2.imshow("Results", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    STOP_ALL_PROCESSES = True
                    break
                
    elif mode_selection == 2: # 싱글 스레드(뭔가 잘 안되는 것 같기도...)
        with mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            cap = cv2.VideoCapture(camera_selection)
            while cap.isOpened():
                success, image = cap.read()  # 한 프레임을 읽습니다.
                if success:  # 프레임을 읽는데 성공했다면
                    results, image, h, w, _ = ImageHandler.processImage(hands, image)
                    if results.multi_hand_landmarks:  # 손(들)을 찾았다면
                        image, K, extended_K, calib_R_t_with_0001, dist = ImageHandler.getCalibrationMatrices(camera_selection, SIZE)
                        for i, hand_landmarks in enumerate(results.multi_hand_landmarks): # 손(들)을 돌면서
                            # 한손이면 i는 0, 양손이면 i는 0, 1
                            # hand_landmarks는 [노멀라이즈드랜드마크, 노멀라이즈드랜드마크, 노멀라이즈드랜드마크, ... ] 처럼
                            # NormalizedLandmark 여러개를 담은 리스트입니다.(hand_landmark's' 이기떄문에 복수형)
                            # Python의 List<NormalizedLandmarkList>
                            
                            model_points = np.float32([[l.x, l.y, l.z] for l in results.multi_hand_world_landmarks[i].landmark])  # Hand 3D 좌표
                            image_points = np.float32([[l.x * w, l.y * h] for l in hand_landmarks.landmark])  # Hand 2D 좌표

                            # H2C에서 H곱해서 나온 C좌표를 2D이미지로 찍을때 어떤 픽셀에 찍어야 할지 리스트와
                            # C2W에서 C곱해서 나온 W좌표를 2D이미지로 찍을때 어떤 픽셀에 찍어야 할지 리스트
                            # 두개를 리턴받습니다.
                            h2c2i_coords, w2c2i_coords = MatrixHandler.getCoordinates(K, extended_K, dist, calib_R_t_with_0001, model_points, image_points)

                            # 시각적 분석
                            image = Analyzer.visualAnalyze(image_points, h2c2i_coords, w2c2i_coords, h, w, image)

                    cv2.imshow("Results", image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cv2.destroyAllWindows()
                        STOP_ALL_PROCESSES = True
                        break
                    
    elif mode_selection == 3: # 1회성 캘리브레이션
        with mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            cap = cv2.VideoCapture(camera_selection)
            while cap.isOpened():
                success, image = cap.read()  # 한 프레임을 읽습니다.
                if success:  # 프레임을 읽는데 성공했다면
                    results, image, h, w, _ = ImageHandler.processImage(hands, image)
                    if results.multi_hand_landmarks:  # 손(들)을 찾았다면
                        for i, hand_landmarks in enumerate(results.multi_hand_landmarks): # 손(들)을 돌면서
                            # 한손이면 i는 0, 양손이면 i는 0, 1
                            # hand_landmarks는 [노멀라이즈드랜드마크, 노멀라이즈드랜드마크, 노멀라이즈드랜드마크, ... ] 처럼
                            # NormalizedLandmark 여러개를 담은 리스트입니다.(hand_landmark's' 이기떄문에 복수형)
                            # Python의 List<NormalizedLandmarkList>
                            
                            model_points = np.float32([[l.x, l.y, l.z] for l in results.multi_hand_world_landmarks[i].landmark])  # Hand 3D 좌표
                            image_points = np.float32([[l.x * w, l.y * h] for l in hand_landmarks.landmark])  # Hand 2D 좌표

                            # H2C에서 H곱해서 나온 C좌표를 2D이미지로 찍을때 어떤 픽셀에 찍어야 할지 리스트와
                            # C2W에서 C곱해서 나온 W좌표를 2D이미지로 찍을때 어떤 픽셀에 찍어야 할지 리스트
                            # 두개를 리턴받습니다.
                            h2c2i_coords, w2c2i_coords = MatrixHandler.getCoordinates(K, extended_K, dist, calib_R_t_with_0001, model_points, image_points)

                            # 시각적 분석
                            image = Analyzer.visualAnalyze(image_points, h2c2i_coords, w2c2i_coords, h, w, image)

                    cv2.imshow("Results", image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cv2.destroyAllWindows()
                        STOP_ALL_PROCESSES = True
                        break

def initialize():
    STOP_ALL_PROCESSES = False
    SIZE = tuple(map(int, input("체스보드 사이즈 입력 (ex. 9 7) > ").split()))
    camera_selection = int(input("카메라 선택 > "))
    mode_selection = int(input("모드를 선택하세요(1: 멀티스레드, 2: 싱글스레드, 3: 1회성 캘리브레이션, 4: 멀티프로세스) > "))
    return STOP_ALL_PROCESSES, SIZE, camera_selection, mode_selection, *ImageHandler.getCalibrationMatrices(camera_selection, SIZE)


STOP_ALL_PROCESSES, SIZE, camera_selection, mode_selection, global_image, K, extended_K, calib_R_t_with_0001, dist = initialize()

if mode_selection == 1: # 멀티 스레드
    if __name__ == "__main__":
        t1 = Thread(target=realTimeEstimation, args=(camera_selection, mode_selection))
        t2 = Thread(target=realTimeCalibration, args=(camera_selection, SIZE))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
elif mode_selection == 2: # 싱글 스레드
    realTimeEstimation(camera_selection, mode_selection)
elif mode_selection == 3: # 1회성 캘리브레이션
    realTimeEstimation(camera_selection, mode_selection)
elif mode_selection == 4: # 멀티 프로세싱(이상함)
    if __name__ == "__main__":
        print("시작")
        p1 = Process(target=realTimeEstimation, args=(camera_selection, mode_selection))
        p2 = Process(target=realTimeCalibration, args=(camera_selection, SIZE))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        
else:
    print("잘못된 선택입니다.")