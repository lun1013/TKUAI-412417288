import cv2
import pytesseract




image = cv2.imread('car3.jpg')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


blurred = cv2.GaussianBlur(gray, (5, 5), 0)


edged = cv2.Canny(blurred, 30, 200)


threshold = cv2.adaptiveThreshold(edged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


possible_plates = []
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    if 2 < aspect_ratio < 6:  
        possible_plates.append((x, y, w, h))


if not possible_plates:
    print("沒有找到可能的車牌。")
    exit(0)


for (x, y, w, h) in possible_plates:
   
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

   
    plate = image[y:y+h, x:x+w]

  
    gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    
    text = pytesseract.image_to_string(gray_plate, config='--psm 7')

   
    print("辨識結果：", text)


cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()