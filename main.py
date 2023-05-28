from hand_pose_tools import *
from multiprocessing import Process
import threading

def multiprocessingCalibration(camera_selection, SIZE):
    global STOP_ALL_PROCESSES, K, extended_K, dist, calib_R_t_with_0001

    while not STOP_ALL_PROCESSES:
        K, extended_K, dist, calib_R_t_with_0001 = ImageHandler.getCalibrationMatrices(camera_selection, SIZE)

def multiprocessingEstimation(camera_selection):
    global thread_selection, STOP_ALL_PROCESSES, global_image, K, extended_K, dist, calib_R_t_with_0001

    with mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands: # 손을 추적할 모델입니다.
        cap = cv2.VideoCapture(camera_selection)
        while cap.isOpened():
            success, global_image = cap.read()  # 한 프레임을 읽습니다.
            if success:  # 프레임을 읽는데 성공했다면
                results, global_image, h, w, _ = ImageHandler.processImage(hands, global_image) # 랜드마크 추적결과, 이미지의 기본 정보(이미지 픽셀별 값, 가로길이, 세로길이)

                # """FOR SINGLE THREAD"""
                # K, extended_K, calib_R_t_with_0001, dist = ImageHandler.getCalibrationMatrices(camera_selection, SIZE)
                # """DONE"""

                if results.multi_hand_landmarks:  # 손(들)을 찾았다면
                    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):  # 한손이면 i는 0, 양손이면 i는 0, 1
                        # hand_landmarks는 [노멀라이즈드랜드마크, 노멀라이즈드랜드마크, 노멀라이즈드랜드마크, ... ] 처럼
                        # NormalizedLandmark 여러개를 담은 리스트입니다.(hand_landmark's' 이기떄문에 복수형)
                        # Python의 List<NormalizedLandmarkList>
                        
                        model_points = np.float32([[l.x, l.y, l.z] for l in results.multi_hand_world_landmarks[i].landmark])  # 손 3D 좌표
                        image_points = np.float32([[l.x * w, l.y * h] for l in hand_landmarks.landmark])  # 손 2D 좌표

                        # H2C에서 H곱해서 나온 C좌표를 2D이미지로 찍을때 어떤 픽셀에 찍어야 할지 리스트
                        # C2W에서 C곱해서 나온 W좌표를 2D이미지로 찍을때 어떤 픽셀에 찍어야 할지 리스트
                        # 두개를 리턴받습니다.
                        
                        


                        h2c2i_coords, w2c2i_coords = MatrixHandler.getCoordinates(K, extended_K, dist, calib_R_t_with_0001, model_points, image_points)

                        # 수치적, 시각적 분석
                        global_image = Analyzer.analyzeResult(image_points, h2c2i_coords, w2c2i_coords, h, w, global_image)

                cv2.imshow("Results", global_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    STOP_ALL_PROCESSES = True
                    break


if __name__ == "__main__":
    STOP_ALL_PROCESSES, SIZE = False, tuple(map(int, input("체스보드 사이즈 입력 (ex. 9 7) > ").split()))
    camera_selection = int(input("카메라 선택 > "))
    thread_selection = int(input("모드를 선택하세요(1: 싱글스레드, 2: 멀티스레드, 3: 멀티프로세싱) > "))

    global_image, K, extended_K, calib_R_t_with_0001, dist = None, *ImageHandler.getCalibrationMatrices(camera_selection, SIZE)

    if thread_selection == 1: # 단일 스레드
        multiprocessingEstimation(camera_selection)
    elif thread_selection == 2: # thread 로 도전
        t1 = threading.Thread(target=multiprocessingEstimation, args=(camera_selection, ))
        t1.start()
        t2 = threading.Thread(target=multiprocessingCalibration, args=(camera_selection, SIZE))
        t2.start()
        t1.join()
        t2.join()
    elif thread_selection == 3: # Process 로 도전
        p1 = Process(target=multiprocessingEstimation, args=(camera_selection, ))
        p1.start()
        p2 = Process(target=multiprocessingCalibration, args=(camera_selection, SIZE))
        p2.start()
        p1.join()
        p2.join()
    else:
        print("잘못된 선택입니다.")
    
    
    
    
    

