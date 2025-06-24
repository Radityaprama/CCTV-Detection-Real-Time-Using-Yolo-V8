import cv2

video_source = "https://pelindung.bandung.go.id:3443/video/HIKSVISION/Soekar.m3u8"

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("❌ Gagal buka video source!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal ambil frame")
        break

    cv2.imshow("🎥 Tes Video", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        print("Keluar...")
        break

cap.release()
cv2.destroyAllWindows()
