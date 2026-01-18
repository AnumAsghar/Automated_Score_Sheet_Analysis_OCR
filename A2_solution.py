import cv2
import numpy as np
import os
from paddleocr import PaddleOCR
from statistics import median
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math

@dataclass
class Cell:
    x: int
    y: int
    w: int
    h: int
    content: str = ""
    score: float = 0.0
    confidence: float = 0.0
    is_numeric: bool = False

class ScoreSheetAnalyzer:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            rec_algorithm='SVTR_LCNet',
            det_algorithm='DB',
            rec_batch_num=1,
            det_limit_side_len=2000,
            det_db_thresh=0.25,
            det_db_box_thresh=0.25,
            det_db_unclip_ratio=1.75,
            rec_image_shape="3, 32, 320"
        )
        
        self.BORDER_COLOR = (0, 255, 0)
        self.TEXT_COLOR = (0,210,0)
        self.BORDER_THICKNESS = 2
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.5
        self.FONT_THICKNESS = 2
        self.MAX_SCORE = 45
        self.EXPECTED_COLS = 7
        self.EXPECTED_ROWS = 3
        self.OCR_ATTEMPTS = 3

    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        # Corrects image skew using line detection
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
            
            if lines is not None:
                angles = []
                for rho, theta in lines[:, 0]:
                    if abs(theta - np.pi/2) < np.pi/4:
                        angle = theta - np.pi/2
                        angles.append(angle)
                
                if angles:
                    median_angle = np.median(angles)
                    angle_degrees = median_angle * 180/np.pi
                    
                    if abs(angle_degrees) > 0.5:
                        height, width = image.shape[:2]
                        center = (width//2, height//2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
                        rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                               flags=cv2.INTER_CUBIC,
                                               borderMode=cv2.BORDER_REPLICATE)
                        return rotated
            return image
        except:
            return image

    def is_numeric_cell(self, row_idx: int, cell_idx: int) -> bool:
        # Checks if cell should contain numeric value based on position
        return row_idx == self.EXPECTED_ROWS - 1 and cell_idx > 0

    def preprocess_cell(self, cell_image: np.ndarray) -> List[np.ndarray]:
        # Prepares cell image for OCR with multiple preprocessing variations
        preprocessed = []
        preprocessed.append(cell_image)
        
        scale_factor = 2
        enlarged = cv2.resize(cell_image, None, 
                            fx=scale_factor, fy=scale_factor, 
                            interpolation=cv2.INTER_CUBIC)
        preprocessed.append(enlarged)
        
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR))
        
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        preprocessed.append(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR))
        
        return preprocessed

    def recognize_content(self, cell_image: np.ndarray, is_numeric: bool = False, is_score_row: bool = False) -> Tuple[str, float]:
        # Recognizes cell content with optimized settings for score row
        best_text = ""
        best_confidence = 0.0
        MIN_CONFIDENCE = 0.3 if is_score_row else 0.5
        
        preprocessed_images = self.preprocess_cell(cell_image)
        
        for prep_image in preprocessed_images:
            try:
                result = self.ocr.ocr(prep_image, cls=not is_score_row)
                if result and result[0]:
                    for line in result[0]:
                        text = line[1][0].strip()
                        confidence = line[1][1]
                        
                        if is_numeric:
                            text = text.lower()
                            text = text.replace('o', '0').replace('l', '1').replace('i', '1')
                            text = text.replace('s', '5').replace('b', '8')
                            text = ''.join(c for c in text if c.isdigit() or c in '.,')
                            
                            if text and confidence >= MIN_CONFIDENCE:
                                try:
                                    text = text.replace(',', '.')
                                    num = float(text)
                                    if is_score_row:
                                        if num > 10 and num < 100:
                                            num /= 10
                                        elif num < 1:
                                            num *= 10
                                        if 0 <= num <= 10:
                                            best_text = f"{num:.1f}"
                                            best_confidence = confidence
                                    else:
                                        if 0 <= num <= 10 and confidence > best_confidence:
                                            best_text = f"{num:.1f}"
                                            best_confidence = confidence
                                except ValueError:
                                    continue
                        else:
                            if confidence > best_confidence:
                                best_text = text
                                best_confidence = confidence
            except:
                continue
        
        return best_text, best_confidence

    def draw_text_in_cell(self, image: np.ndarray, text: str, cell: Cell, is_total: bool = False) -> None:
        # Draws text centered in cell 
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)
        
        text_x = cell.x + (cell.w - text_width) // 2
        text_y = cell.y + (cell.h + text_height) // 2
        
        color = self.TEXT_COLOR
        cv2.putText(image, text, (text_x, text_y), self.FONT, 
                    self.FONT_SCALE, color, self.FONT_THICKNESS)

    def extract_table_structure(self, image: np.ndarray) -> List[List[Cell]]:
        # Extracts table structure and cells from image
        deskewed = self.deskew_image(image)
        gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        img_height, img_width = binary.shape
        horizontal_kernel_size = max(img_width // 30, 40)
        vertical_kernel_size = max(img_height // 30, 40)
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_size, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_size))
        
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        grid = cv2.add(horizontal_lines, vertical_lines)
        contours, hierarchy = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            median_area = np.median([a for a in areas if a > 0])
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                area = w * h
                
                if (0.2 < aspect_ratio < 5.0 and 
                    area > median_area * 0.1 and 
                    area < median_area * 10 and
                    w > 10 and h > 10):
                    cells.append(Cell(x=x, y=y, w=w, h=h))
        
        rows = []
        if cells:
            cells.sort(key=lambda c: (c.y, c.x))
            row_height = np.median([c.h for c in cells])
            y_tolerance = row_height * 0.5
            
            current_row = []
            prev_y = cells[0].y
            
            for cell in cells:
                if abs(cell.y - prev_y) < y_tolerance:
                    current_row.append(cell)
                else:
                    if len(current_row) >= self.EXPECTED_COLS - 2:
                        current_row.sort(key=lambda c: c.x)
                        rows.append(current_row)
                    current_row = [cell]
                    prev_y = cell.y
            
            if current_row and len(current_row) >= self.EXPECTED_COLS - 2:
                current_row.sort(key=lambda c: c.x)
                rows.append(current_row)
        
        return rows

    def process_image(self, image_path: str) -> bool:
        # Main processing function for score sheet images
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            result_image = image.copy()
            rows = self.extract_table_structure(image)
            
            if not rows:
                output_path = os.path.join(self.output_dir, os.path.basename(image_path))
                cv2.imwrite(output_path, image)
                return False
            
            total_score = 0
            
            for row_idx, row in enumerate(rows):
                is_score_row = (row_idx == len(rows) - 1)
                for cell_idx, cell in enumerate(row):
                    try:
                        cell_image = image[cell.y:cell.y+cell.h, cell.x:cell.x+cell.w]
                        is_total_cell = (is_score_row and cell_idx == len(row) - 1)
                        is_numeric = self.is_numeric_cell(row_idx, cell_idx)
                        
                        cv2.rectangle(result_image, (cell.x, cell.y),
                                    (cell.x+cell.w, cell.y+cell.h),
                                    self.BORDER_COLOR,
                                    self.BORDER_THICKNESS)
                        
                        if is_score_row and cell_idx > 0:
                            content, confidence = self.recognize_content(cell_image, True, True)
                            if cell_idx < len(row) - 1:
                                try:
                                    if content:
                                        score = float(content)
                                        if 0 <= score <= 10:
                                            total_score += score
                                except ValueError:
                                    continue
                            else:
                                total_score = min(total_score, self.MAX_SCORE)
                                content = f"{total_score:.1f}"
                        else:
                            content, confidence = self.recognize_content(cell_image, is_numeric, False)
                        
                        if content:
                            self.draw_text_in_cell(result_image, content, cell, is_total_cell)
                    except:
                        continue
            
            output_path = os.path.join(self.output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, result_image)
            return True
            
        except:
            try:
                output_path = os.path.join(self.output_dir, os.path.basename(image_path))
                cv2.imwrite(output_path, image)
            except:
                pass
            return False

def main():
    analyzer = ScoreSheetAnalyzer()
    dataset_dir = "dataset"
    
    if not os.path.exists(dataset_dir):
        return
    
    successful = 0
    total = 0
    
    for i in range(1, 57):
        image_path = os.path.join(dataset_dir, f"image{i}.png")
        if os.path.exists(image_path):
             analyzer.process_image(image_path)


if __name__ == "__main__":
    main()