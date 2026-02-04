import cv2
import numpy as np
import os # パス操作用に追加

def main(args=None):
    # --- 画像のパス処理 ---
    # 画像が実行場所になくても読み込めるよう、ファイルと同じ場所に画像を置く前提でパスを作成
    file_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(file_dir, "ball1.webp")

    # 画像読み込み
    img = cv2.imread(img_path)
    
    # 画像が見つからなかった場合のエラー処理
    if img is None:
        print(f"Error: 画像が見つかりません: {img_path}")
        return

    # HSV変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- 青色の範囲 ---
    lower_blue = np.array()
    upper_blue = np.array()
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # --- 赤色の範囲（2つ必要） ---
    lower_red1 = np.array()
    upper_red1 = np.array([2])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array()
    upper_red2 = np.array()
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # 赤のマスクを合成
    mask_red = mask_red1 | mask_red2

    # --- 青色の輪郭処理 ---
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_blue:
        area = cv2.contourArea(cnt)
        if area > 500:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(img, center, radius, (255, 0, 0), 2)
            cv2.putText(img, "Blue", (center - 20, center[3] - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # --- 赤色の輪郭処理 ---
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        if area > 500:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(img, center, radius, (0, 0, 255), 2)
            cv2.putText(img, "Red", (center - 20, center[3] - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 結果表示
    cv2.imshow("Detected Balls", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()