import cv2
import numpy as np
import os
def detect_potholes_debug(image_path, output_path="detected_potholes.jpg", resize_max=1200, save_debug=True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("cv2.imread returned None (cannot read image).")
    h0, w0 = img.shape[:2]
    if max(h0, w0) > resize_max:
        scale = resize_max / float(max(h0, w0))
        img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_dark = cv2.inRange(hsv[:,:,2], 0, 100)
    mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    edges = cv2.Canny(gray, 80, 160)
    edges_dil = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    combined = cv2.bitwise_and(mask_dark, edges_dil)
    if save_debug:
        base = os.path.splitext(output_path)[0]
        cv2.imwrite(base + "_darkmask.png", mask_dark)
        cv2.imwrite(base + "_edges.png", edges)
        cv2.imwrite(base + "_combined.png", combined)
    contours, _ = cv2.findContours(mask_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    potholes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-8)
        region_vals = gray[mask == 255]
        variance = np.var(region_vals) if len(region_vals) > 0 else 0
        edge_density = np.sum(edges[mask == 255]) / (area + 1e-8)
        mean_val = np.mean(region_vals) if len(region_vals) > 0 else 255
        if circularity > 0.85:  
            continue
        if edge_density < 0.02: 
            continue
        if variance < 20: 
            continue
        if mean_val > 130:
            continue
        radius = min(max(w, h) // 2, 100)
        potholes.append((x, y, w, h, radius))
    out = img.copy()
    for (x, y, w, h, r) in potholes:
        cx, cy = x + w//2, y + h//2
        cv2.circle(out, (cx, cy), r, (0, 0, 255), 2)
    cv2.imwrite(output_path, out)
    print(f"Detected {len(potholes)} potholes.")
    return output_path, potholes
if __name__ == "__main__":
    input_file = "images.jpeg"
    out_file = "detected_potholes.jpg"
    out_path, potholes = detect_potholes_debug(input_file, out_file, save_debug=True)
    print("Output saved to:", out_path)
    print("Potholes:", potholes)
