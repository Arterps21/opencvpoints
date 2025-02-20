import cv2
import numpy as np


def main():
    # Загрузка цветных изображений
    distorted = cv2.imread('754.jpg', cv2.IMREAD_COLOR)
    reference = cv2.imread('7541.png', cv2.IMREAD_COLOR)

    if distorted is None or reference is None:
        print("Ошибка загрузки изображений!")
        return

    # Конвертация в градации серого для обработки
    gray_dist = cv2.cvtColor(distorted, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # Настройка детектора SIFT
    sift = cv2.SIFT_create(
        contrastThreshold=0.02,
        edgeThreshold=15
    )

    # Поиск ключевых точек
    kp1, des1 = sift.detectAndCompute(gray_dist, None)
    kp2, des2 = sift.detectAndCompute(gray_ref, None)


    # Фильтрация ключевых точек
    min_keypoint_size = 5
    kp1 = [kp for kp in kp1 if kp.size > min_keypoint_size]
    kp2 = [kp for kp in kp2 if kp.size > min_keypoint_size]


    # Обновление дескрипторов
    _, des1 = sift.compute(gray_dist, kp1)
    _, des2 = sift.compute(gray_ref, kp2)

    # Сопоставление ключевых точек
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:int(len(matches) * 0.15)]

    print(f"Найдено совпадений: {len(good)}")

    # Визуализация совпадений
    matches_img = cv2.drawMatches(
        distorted, kp1,
        reference, kp2,
        good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0)  # Зеленые линии
    )
    cv2.imwrite("matches_debug.jpg", matches_img)
    print("Визуализация совпадений сохранена!")

    if len(good) < 10:
        print("Недостаточно совпадений!")
        return

    # Вычисление гомографии
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000
    )

    if M is None or np.isnan(M).any():
        print("Ошибка в матрице преобразования!")
        return

    # Применение преобразования
    restored_color = cv2.warpPerspective(
        distorted,
        M,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LANCZOS4
    )

    # Сохранение результатов
    cv2.imwrite('restored_color.jpg', restored_color)
    print("Восстановленное изображение сохранено!")


if __name__ == "__main__":
    main()