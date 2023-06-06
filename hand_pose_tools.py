import cv2, numpy as np, mediapipe as mp

class ImageHandler:
    def getCalibrationMatrices(camera_selection, SIZE): # 캘리브레이션        
        # 내부파라미터, 왜곡, 캘리브레이션_rvec, 캘리브레이션_tvec
        image, _, K, dist, calib_rvecs, calib_tvecs = ImageHandler.calibrateMyCamera(camera_selection, SIZE)
        # 3x3 내부파라미터를 3x4 로 확장합니다.
        extended_K = MatrixHandler.extendMatrix("K", K)
        # 캘리브레이션_rvec 와 캘리브레이션_tvec 이용하여 4x4 변환행렬을 만듭니다.
        calib_R_t_with_0001 = MatrixHandler.extendMatrix("[R|t]", calib_rvecs[0], calib_tvecs[0])
        return image, K, extended_K, calib_R_t_with_0001, dist # 내부파라미터, 확장내부파라미터, 변환행렬, 왜곡
    
    def calibrateMyCamera(camera_selection, SIZE):
        cap = cv2.VideoCapture(camera_selection)
        while cap.isOpened():
            success, image = cap.read()
            if success:
                image = cv2.flip(image, 1)

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                object_points = np.zeros((1, SIZE[0] * SIZE[1], 3), np.float32)
                object_points[0, :, :2] = np.mgrid[0 : SIZE[0], 0 : SIZE[1]].T.reshape(-1, 2)

                success, corners = cv2.findChessboardCorners(gray, SIZE, None)
            if success:
                print("Corner 찾기 성공!")
                return image, *cv2.calibrateCamera([object_points], [corners], gray.shape[::-1], None, None)
            print("Corner 검출 실패")

    def processImage(hands, image): # 이미지 속성 및 랜드마크 분석결과 가져오기
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        h, w, d = image.shape
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return results, image, h, w, d
            
class MatrixHandler:
    def extendMatrix(mode, *matrices): # 행렬을 확장합니다.
        if mode == "[R|t]": # 3x4 R|t 행렬을 4x4 행렬로 확장합니다.
            rvecs, tvecs = matrices
            R, _ = cv2.Rodrigues(rvecs)
            col_tvec = np.array([tvecs[0], tvecs[1], tvecs[2]])
            R_t = np.append(R, col_tvec, axis=1)
            R_t_with_0001 = np.append(R_t, [np.array([0, 0, 0, 1])], axis=0)
            return R_t_with_0001
        elif mode == "K": # 3x3 K 행렬을 3x4 행렬로 확장합니다.
            return np.append(matrices[0], np.array([[0], [0], [0]]), axis=1)
    
    def scaleLastElementAs1(matrix): # [sx, sy, s] -> s[x, y, 1] -> [x, y, 1] 즉, 마지막 원소를 1로 만들어 s(scaling)을 없앱니다.
        for line in matrix: line /= line[2]
        return matrix

    def getCoordinates(K, extended_K, dist, calib_R_t_with_0001, model_points, image_points): 
        # 캘리브레이션과 results.multi_hand_landmarks에서 구한 값들을 통해 solvePnP를 돌리고
        # 그 rvec, tvec을 요리조리 조작하여 최종적으로 H2C_2D좌표, W2C_2D좌표를 리턴합니다.

        # H2C 구하기
        _, solvepnp_rvecs, solvepnp_tvecs = cv2.solvePnP(model_points, image_points, K, dist, flags=cv2.SOLVEPNP_EPNP)
        H2C_MATRIX = MatrixHandler.extendMatrix("[R|t]", solvepnp_rvecs, solvepnp_tvecs)

        # H2C * H -> C
        h2c_coords = [H2C_MATRIX @ np.array([X, Y, Z, 1]) for X, Y, Z in model_points]
    
        # W2C 구하기 from calib_rvecs, calib_tvecs
        W2C_MATRIX = calib_R_t_with_0001

        # W2C -> C2W 구하기
        C2W_MATRIX = np.linalg.inv(W2C_MATRIX)

        # C2W * C -> W
        c2w_coords = [C2W_MATRIX @ np.array([X, Y, Z, 1]) for X, Y, Z, _ in h2c_coords]

        # W2C * W -> C
        w2c_coords = [W2C_MATRIX @ np.array([X, Y, Z, 1]) for X, Y, Z, _ in c2w_coords]


        # 3D -> 2D 바꾼다음, 그 2D를 scaling 제거해서 순수한 카메라좌표계의 x, y 를 리턴합니다.
        h2c2i_coords = MatrixHandler.scaleLastElementAs1([extended_K @ coord for coord in h2c_coords])
        w2c2i_coords = MatrixHandler.scaleLastElementAs1([extended_K @ coord for coord in w2c_coords])

        return h2c2i_coords, w2c2i_coords

class Analyzer:
    def visualAnalyze(image_points, h2c2i_coords, w2c2i_coords, h, w, image=None): # h2c2i: hand_to_camera_to_image, w2c2i: world_to_camera_to_image
        # 1. H2C와 W2C 원(circle) 그리기
        for landmark_number in range(21):
            h2c_x, h2c_y, _ = map(int, h2c2i_coords[landmark_number])
            w2c_x, w2c_y, _ = map(int, w2c2i_coords[landmark_number])

            if 0 <= h2c_x*w2c_x <= w*w and 0 <= h2c_y*w2c_y <= h*h:
                image = cv2.circle(image, (h2c_x, h2c_y), 6, (0, 255, 0), 2)
                image = cv2.circle(image, (w2c_x, w2c_y), 10, (255, 0, 0), 2)
        
        # 2. H2C와 W2C 의 관절 사이 선분(line) 그리기
        for start_landmark_number, end_landmark_number in [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
                                                           (9, 10), (10, 11), (11, 12), (13, 14), (14, 15),
                                                           (15, 16), (17, 18), (18, 19), (19, 20), (0, 1),
                                                           (0, 5), (0, 17), (5, 9), (9, 13), (13, 17)]:
            
            # H2C
            h2c_1_x, h2c_1_y, _ = map(int, h2c2i_coords[start_landmark_number])
            h2c_2_x, h2c_2_y, _ = map(int, h2c2i_coords[end_landmark_number])

            if 0 <= h2c_1_x*h2c_2_x <= w*w and 0 <= h2c_1_y*h2c_2_y <= h*h:
                image = cv2.line(image, (h2c_1_x, h2c_1_y), (h2c_2_x, h2c_2_y), (0, 255, 0), 2)

            # W2C
            w2c_1_x, w2c_1_y, _ = map(int, w2c2i_coords[start_landmark_number])
            w2c_2_x, w2c_2_y, _ = map(int, w2c2i_coords[end_landmark_number])

            if 0 <= w2c_1_x*w2c_2_x <= w*w and 0 <= w2c_1_y*w2c_2_y <= h*h:
                image = cv2.line(image, (w2c_1_x, w2c_1_y), (w2c_2_x, w2c_2_y), (255, 0, 0), 2)

        return image
    
    def numericAnalyze(image_points, h2c2i_coords, w2c2i_coords):
        for landmark_number in range(21):
            mp_x, mp_y = image_points[landmark_number]
            h2c_x, h2c_y, _ = h2c2i_coords[landmark_number]
            w2c_x, w2c_y, _ = w2c2i_coords[landmark_number]
            print("-------------------------------------------------------")
            print("Pure MediaPipe  Point %d: [%.7f, %.7f]" % (landmark_number+1, mp_x, mp_y))
            print("H2C_to_Image    Point %d: [%.7f, %.7f]" % (landmark_number+1, h2c_x, h2c_y))
            print("W2C_to_Image    Point %d: [%.7f, %.7f]" % (landmark_number+1, w2c_x, w2c_y))
