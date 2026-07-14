# Functions for solving sudoku puzzles

import cv2
import imutils
import os
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sudoku_solver_class import SudokuSolver


def resize_and_maintain_aspect_ratio(input_image, new_width):
    orig_width, orig_height = input_image.shape[1], input_image.shape[0]
    ratio = new_width / float(orig_width)
    new_height = int(orig_height * ratio)
    dim = (new_width, new_height)
    reshaped_image = cv2.resize(input_image, dim, interpolation=cv2.INTER_AREA)
    
    return reshaped_image


def apply_grayscale_blur_and_threshold(img, method="mean", blocksize=91, c=7):
    img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if method == "mean":
        adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C
    elif method == "gaussian":
        adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh = cv2.adaptiveThreshold(gray,
                                    maxValue=255,
                                    adaptiveMethod=adaptiveMethod,
                                    thresholdType=cv2.THRESH_BINARY,
                                    blockSize=blocksize,
                                    C=c)

    thresh = cv2.bitwise_not(thresh)

    return thresh


def get_quadrilateral_points_in_order(approx_arr):

    
    try:
        assert(approx_arr.shape == (4, 1, 2) or approx_arr.shape == (4, 2))
    except:
        raise ValueError(f"Incorrect shape for approx_arr: {approx_arr.shape}. Requires shape of (4, 1, 2) or (4, 2).")

    if approx_arr.shape == (4, 1, 2):
        approx_arr = np.squeeze(approx_arr, axis=1)
    
    max_x = int(1.1 * np.max(approx_arr[:,0]))
    origin_1 = [0, 0]
    origin_2 = [max_x, 0]
    distances_1 = [np.linalg.norm(point - origin_1) for point in approx_arr]
    distances_2 = [np.linalg.norm(point - origin_2) for point in approx_arr]
    tl_idx = np.argmin(distances_1)
    br_idx = np.argmax(distances_1)

    dist_arr = distances_2.copy()
    dist_arr[tl_idx] = np.inf
    dist_arr[br_idx] = np.inf
    tr_idx = np.argmin(dist_arr)

    dist_arr = distances_2.copy()
    dist_arr[tl_idx] = -np.inf
    dist_arr[br_idx] = -np.inf
    bl_idx = np.argmax(dist_arr)
    tl = approx_arr[tl_idx]
    br = approx_arr[br_idx]
    tr = approx_arr[tr_idx]
    bl = approx_arr[bl_idx]
    
    return np.array([tl, tr, br, bl])


def perform_four_point_transform(input_img, src_corners, pad=10):
    '''
    Perform a four-point perspective transform to an image such that the four
    corners in img are mapped to new specified points in the resulting image. 

    Args:
        input_img: An image array.
        src_corners: An array of shape (4, 2) containing the (x, y) locations
                     of the four reference points in input_img.
        pad: Pixel value for padding applied to all sides of warped image.
             The warped image then contains pixels extending past the corners,
             which is useful for accommodating curves in the paper surface for
             locating the puzzle grid later.
    
    Returns:
        M: transformation matrix.
        warped: Image resulting from the perspective transform applied to
                input_img.
    
    '''
    
    # Get the corner points in the order we want
    src_corners = get_quadrilateral_points_in_order(src_corners)
    src_corners = src_corners.astype("float32")
    tl, tr, br, bl = src_corners
    
    # Define the desired dimensions of the destination (warped) image (max_width, max_height)
    # Calculate the width of the top and bottom edges, and take the maximum
    bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(bottom_width), int(top_width))

    # Calculate the height of the left and right edges, and take the maximum
    left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_height = max(int(left_height), int(right_height))

    # Define the set of points in the destination image that our four corners 
    # from the input image should map to, such that we obtain a bird's eye view
    dest_img_corners = np.array([[0+pad, 0+pad],
                                 [max_width-1-pad, 0+pad],
                                 [max_width-1-pad, max_height-1-pad],
                                 [0+pad, max_height-1-pad]], dtype="float32")

    # Compute our transformation matrix
    M = cv2.getPerspectiveTransform(src=src_corners, dst=dest_img_corners)
    warped_img = cv2.warpPerspective(input_img, M, (max_width, max_height))

    return M, warped_img


def find_grid_contour_candidates(img, to_plot=False):
    '''
    Given an input image, build a list of contours that may represent the
    main puzzle grid outline. For each candidate identified, perform a
    perspective transformation and store the contour, the transformation
    matrix, and the perspective-warped image.
    
    Args:
        img: RGB image of sudoku puzzle.
        to_plot: Flag to produce plot showing all contours.
        
    Returns:
        contours: A list of contours identified as potential grid candidates.
                    Sorted in reverse order by area of contour.
        M: A list of perspective transformation matrices.
        warped: A list of warped images that may represent the main puzzle grid.
        
    '''
    
    # Initialise lists for return values
    M_matrices = []
    warped_images = []
    contour_grid_candidates = []

    # Calculate the area of the whole image
    img_area = img.shape[0] * img.shape[1]

    # Apply blur, grayscale and adaptive threshold
    thresh = apply_grayscale_blur_and_threshold(img, blocksize=41, c=8)

    # Extract edges with Canny as an additional contour source.
    # Adaptive-threshold contours are retained because they are often more
    # reliable when the grid lines are already closed.
    edges = cv2.Canny(thresh, threshold1=50, threshold2=150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours = []
    for source_img in (thresh, edges):
        detected = cv2.findContours(image=source_img.copy(),
                                    mode=cv2.RETR_EXTERNAL,
                                    method=cv2.CHAIN_APPROX_SIMPLE)
        detected = imutils.grab_contours(detected)
        contours.extend(detected)

    if contours:
        # Sort the contours according to contour area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        if to_plot:
            # Plot all the contours on the original RGB image
            with_contours = cv2.drawContours(img.copy(), contours, -1, (0, 255, 75), thickness=2)
            plt.imshow(with_contours)
            plt.show(block=False)
        
        for contour in contours:
            # Approximate the contour in order to determine whether the contour is a quadrilateral
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            
            # Calculate the area of the identified contour
            contour_area = cv2.contourArea(contour)
            contour_fractional_area = contour_area / img_area
            
            # We are looking for a quadrilateral contour with sufficiently large area
            if len(approx) == 4 and contour_fractional_area > 0.1:
                # Get the corners in the required order
                approx = get_quadrilateral_points_in_order(approx)
                # Use the approximation to apply the perspective transform
                # on the candidate grid region
                M, warped_img = perform_four_point_transform(input_img=img,
                                                              src_corners=approx,
                                                              pad=30)
                
                M_matrices.append(M)
                warped_images.append(warped_img)
                contour_grid_candidates.append(contour)
                        
    if warped_images:
        return M_matrices, warped_images, contour_grid_candidates
    else:
        raise Exception("No grid contour candidates were found in image")


def check_for_digit_in_cell_image(img, area_threshold=5, apply_border=False):
    '''
    Determine whether or not a digit is present in an image of a single
    sudoku cell. If a contour is located whose area exceeds area_threshold,
    it is determined that this contour represents a digit.
    
    Args:
        img: An image of a single sudoku cell.
        area_threshold: Threshold value whose units are percentage of image area.
                        A contour is considered a digit if its area exceeds
                        the threshold.
                        
        apply_border: Whether or not to apply a mask to remove non-digit pixels
                      around the edges of the cell image.
            
    Returns:
        image_contains_digit: Boolean - whether the cell is considered to contain a digit.
        cell_img: Image of the sudoku cell, optionally with border masked out.
    
    '''
    
    cell_img = img.copy()
    
    if apply_border:
        # Crude way to eliminate the unwanted pixels around the borders
        border_fraction = 0.07
        replacement_val = 0
        
        y_border_px = int(border_fraction * cell_img.shape[0])
        x_border_px = int(border_fraction * cell_img.shape[1])
        
        cell_img[:, 0:x_border_px] = replacement_val
        cell_img[:, -x_border_px:] = replacement_val
        cell_img[0:y_border_px, :] = replacement_val
        cell_img[-y_border_px:, :] = replacement_val
    
    # Get the contours for the image
    contours = cv2.findContours(image=cell_img,
                                mode=cv2.RETR_TREE,
                                method=cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    if len(contours) > 0:
        # Sort the contours according to contour area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)        
        largest_contour_area = cv2.contourArea(contours[0])
        image_area = cell_img.shape[0] * cell_img.shape[1]
        contour_percentage_area = 100 * largest_contour_area / image_area
        
        if contour_percentage_area > area_threshold:
            image_contains_digit = True
        else:
            image_contains_digit = False
        
    else:
        image_contains_digit = False
        
    return image_contains_digit, cell_img


def locate_cells_within_grid(grid_img, to_plot=False):
    '''
    Identify the individual sudoku cells contained within the grid image.
    
    Args:
        grid_img:
        to_plot:
            
    Returns:
        valid_cells:
    
    '''
    
    # Initialise a list to store the detected cells
    valid_cells = []
    
    # Calculate the area of the sudoku grid.
    # Used later to check if a contour is a valid cell
    grid_area = grid_img.shape[0] * grid_img.shape[1]
    
    # Apply blur, grayscale and adaptive threshold
    grid_img = apply_grayscale_blur_and_threshold(grid_img, method="mean", blocksize=91, c=7)
    
    if to_plot:
        fig, ax = plt.subplots()
        ax.imshow(grid_img, cmap='gray')
        plt.show(block=False)
    
    # Get external and internal contours from the sudoku grid image
    contours = cv2.findContours(image=grid_img.copy(),
                                mode=cv2.RETR_TREE,
                                method=cv2.CHAIN_APPROX_NONE)
    
    if contours:
        # Convenience function to extract contours
        contours = imutils.grab_contours(contours)
        # Sort the contours according to contour area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            # Approximate the contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            # Find the contour area wrt the grid area
            contour_fractional_area = cv2.contourArea(contour) / grid_area
            
            # We are looking for a contour that is quadrilateral and has an area
            # of approximately 1% of the grid area (0.5% to 1.5% works well)
            if len(approx) == 4 and contour_fractional_area > 0.005 and contour_fractional_area < 0.015:
                # We have found a valid contour
                # Use a mask to extract the identified cell from grid_img
                mask = np.zeros_like(grid_img)
                # Use the contour to mask out the cell
                cv2.drawContours(image=mask,
                                contours=[contour],
                                contourIdx=0,
                                color=255,
                                thickness=cv2.FILLED)
                
                # Get the indices where the mask is white
                y_px, x_px = np.where(mask==255)
                # Use these indices to crop out the cell from the image
                cell_image = grid_img[min(y_px):max(y_px)+1, min(x_px):max(x_px)+1]
                # Determine whether or not there's a digit present in the cell
                digit_is_present, cell_image = check_for_digit_in_cell_image(img=cell_image,
                                                                             area_threshold=5,
                                                                             apply_border=True)
                
                # See if erosion improves predictions
                kernel = np.ones((3, 3), np.uint8)
                cell_image = cv2.erode(cell_image, kernel, iterations=1)
                
                # Resize the cell image to be 28x28 pixels for classification later
                cell_image = cv2.resize(cell_image, dsize=(28, 28), interpolation=cv2.INTER_AREA)
                
                # Use the contour to calculate the centroid for the cell. This is
                # so we know the cell's location on the grid, so we know where to
                # place the text when entering the puzzle solution
                moments = cv2.moments(contour)
                x_centroid = int(moments['m10'] / moments['m00'])
                y_centroid = int(moments['m01'] / moments['m00'])
    
                # Create a dictionary for every valid cell
                valid_cells.append({'img': cell_image,
                                    'contains_digit': digit_is_present,
                                    'x_centroid': x_centroid,
                                    'y_centroid': y_centroid})
                        
    else:
        print("No valid cells found in image")
    
    return valid_cells

    
class SudokuExtractionError(RuntimeError):
    """Raised when the grid or its 81 cells cannot be extracted."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def analyze_image_quality(img):
    """Return simple, explainable image-quality measurements.

    The thresholds are practical heuristics and can be tuned for a particular
    camera or dataset.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    warnings = []
    if blur_score < 80:
        warnings.append("image appears blurry")
    if brightness < 45:
        warnings.append("image is too dark")
    elif brightness > 210:
        warnings.append("image is overexposed")
    if contrast < 25:
        warnings.append("image has low contrast")

    return {
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "quality_warnings": warnings,
    }


def estimate_grid_rotation(contour):
    """Estimate the top-edge rotation of a quadrilateral grid in degrees."""
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    if len(approx) != 4:
        return None
    tl, tr, _, _ = get_quadrilateral_points_in_order(approx)
    angle = np.degrees(np.arctan2(float(tr[1] - tl[1]), float(tr[0] - tl[0])))
    return round(float(angle), 2)


def _save_rgb_image(path, image):
    """Save an RGB or grayscale image with OpenCV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = image
    if image.ndim == 3:
        output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), output)


def save_cells_montage(cells, path):
    """Save the extracted cell images in rows of nine."""
    if not cells:
        return
    cell_size = 28
    columns = 9
    rows = int(np.ceil(len(cells) / columns))
    canvas = np.zeros((rows * cell_size, columns * cell_size), dtype=np.uint8)
    for i, cell in enumerate(cells):
        row, col = divmod(i, columns)
        cell_img = cell['img']
        if cell_img.ndim == 3:
            cell_img = np.squeeze(cell_img)
        canvas[row * cell_size:(row + 1) * cell_size,
               col * cell_size:(col + 1) * cell_size] = cell_img
    _save_rgb_image(path, canvas)


def append_diagnostic_report(csv_path, image_path, diagnostics):
    """Append one run to a CSV so successful and failed examples can be compared."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image", "status", "reason", "candidate_count", "cells_found",
        "blur_score", "brightness", "contrast", "rotation_angle",
        "quality_warnings", "debug_directory"
    ]
    row = {name: diagnostics.get(name, "") for name in fieldnames}
    row["image"] = str(image_path)
    if isinstance(row.get("quality_warnings"), list):
        row["quality_warnings"] = " | ".join(row["quality_warnings"])
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_diagnostic_json(path, diagnostics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, ensure_ascii=False, indent=2)


def get_valid_cells_from_image(img, debug_dir=None, return_diagnostics=False):
    """Locate the Sudoku grid and extract exactly 81 cells.

    When ``debug_dir`` is supplied, the input, grayscale, threshold, Canny,
    contour candidates, warped candidates and cell montages are saved. A
    structured diagnostics dictionary is returned when
    ``return_diagnostics=True``.
    """
    debug_path = Path(debug_dir) if debug_dir else None
    if debug_path:
        debug_path.mkdir(parents=True, exist_ok=True)

    diagnostics = analyze_image_quality(img)
    diagnostics.update({
        "status": "processing",
        "reason": "",
        "candidate_count": 0,
        "cells_found": 0,
        "rotation_angle": "",
        "debug_directory": str(debug_path) if debug_path else "",
    })

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = apply_grayscale_blur_and_threshold(img, blocksize=41, c=8)
    edges = cv2.Canny(thresh, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    if debug_path:
        _save_rgb_image(debug_path / "01_input.png", img)
        _save_rgb_image(debug_path / "02_grayscale.png", gray)
        _save_rgb_image(debug_path / "03_threshold.png", thresh)
        _save_rgb_image(debug_path / "04_canny_edges.png", edges)

    try:
        M_matrices, warped_images, grid_candidates = find_grid_contour_candidates(img)
    except Exception as exc:
        diagnostics["status"] = "extraction_failed"
        diagnostics["reason"] = "No suitable quadrilateral grid candidate was found."
        if diagnostics["quality_warnings"]:
            diagnostics["reason"] += " Quality warnings: " + ", ".join(diagnostics["quality_warnings"])
        if debug_path:
            save_diagnostic_json(debug_path / "report.json", diagnostics)
        raise SudokuExtractionError(diagnostics["reason"], diagnostics) from exc

    diagnostics["candidate_count"] = len(warped_images)

    if debug_path:
        contour_preview = img.copy()
        cv2.drawContours(contour_preview, grid_candidates, -1, (0, 255, 0), 3)
        _save_rgb_image(debug_path / "05_grid_candidates.png", contour_preview)

    best_cells = []
    best_index = -1

    for i, grid_image in enumerate(warped_images):
        valid_cells = locate_cells_within_grid(grid_image)

        if debug_path:
            _save_rgb_image(debug_path / f"06_warped_candidate_{i + 1}.png", grid_image)
            save_cells_montage(valid_cells, debug_path / f"07_cells_candidate_{i + 1}.png")

        if len(valid_cells) > len(best_cells):
            best_cells = valid_cells
            best_index = i

        if len(valid_cells) == 81:
            valid_cells = sort_cells_into_grid(valid_cells)
            diagnostics["status"] = "extraction_success"
            diagnostics["reason"] = "Grid and all 81 cells were extracted successfully."
            diagnostics["cells_found"] = 81
            diagnostics["rotation_angle"] = estimate_grid_rotation(grid_candidates[i])
            if diagnostics["rotation_angle"] not in (None, "") and abs(diagnostics["rotation_angle"]) > 15:
                diagnostics["quality_warnings"].append("grid has noticeable rotation")
            if debug_path:
                save_cells_montage(valid_cells, debug_path / "08_cells_sorted_9x9.png")
                save_diagnostic_json(debug_path / "report.json", diagnostics)
            result = (valid_cells, M_matrices[i], grid_image)
            if return_diagnostics:
                return (*result, diagnostics)
            return result

    diagnostics["status"] = "extraction_failed"
    diagnostics["cells_found"] = len(best_cells)
    diagnostics["reason"] = (
        f"A grid candidate was found, but the best candidate contained "
        f"{len(best_cells)} cells instead of 81."
    )
    if best_index >= 0:
        diagnostics["rotation_angle"] = estimate_grid_rotation(grid_candidates[best_index])
    if diagnostics["quality_warnings"]:
        diagnostics["reason"] += " Quality warnings: " + ", ".join(diagnostics["quality_warnings"])
    if debug_path:
        save_cells_montage(best_cells, debug_path / "08_best_failed_cells.png")
        save_diagnostic_json(debug_path / "report.json", diagnostics)
    raise SudokuExtractionError(diagnostics["reason"], diagnostics)

def sort_cells_into_grid(cells):
    '''
    Given a list of cells, use the x and y centroid values to arrange the
    cells into the same order as the original puzzle grid (left to right,
    top to bottom). This sorted list is used later to construct a 2D numpy
    array representing the board, which is passed to a solver.
    
    Each cell is a dictionary containing the cell image, and x- and y-centroids.
    
    Args:
        cells: List of dictionary objects, each containing cell information.
    
    Returns:
        sorted_cells_list: List of cell dictionaries, sorted such that the
                            cells are in the same order as the original puzzle
                            grid (left to right, top to bottom).
    
    '''
    
    x_vals = [cell['x_centroid'] for cell in cells]
    y_vals = [cell['y_centroid'] for cell in cells]
    
    # Get a 2D array of all x, y points from the dictionary
    points = np.array([[cell['x_centroid'], cell['y_centroid']] for cell in cells])
    # Sort by x value
    points_sorted = np.array(sorted(points, key=lambda x: x[1]))
    # Reshape to give an array with 9 rows
    rows = np.reshape(points_sorted, (9, 9, 2))
    # Sort by y value for every row in rows
    final = np.array([sorted(row, key=lambda x: x[0]) for row in rows])
    
    # Make sure all value combinations in final are actually present in the dictionary
    final_reshaped = np.reshape(final, (81, 2))
    for i in range(len(x_vals)):
        assert any(np.equal(final_reshaped, [x_vals[i], y_vals[i]]).all(1))
        
    # Find the index of the cell in cells (list) for each point in final_reshaped
    indices = []
    for x, y in final_reshaped:
        x_indices = np.where(np.array(x_vals) == x)
        y_indices = np.where(np.array(y_vals) == y)
        index = np.intersect1d(x_indices, y_indices)[0]
        indices.append(index)
    
    # Sort the cells list according to the indices list
    sorted_cells_list = [cells[idx] for idx in indices]
    return sorted_cells_list


def plot_cell_images_in_grid(cells):
    '''
    A quick utility function to take all the cell images from a list and plot
    them in a grid in the same layout as the original puzzle.
    
    Args:
        cells: List of dictionary objects, each containing one cell image.
    
    '''
    # Create a blank image (cell images have side 28px)
    width, height = 9*28, 9*28
    main_img = np.zeros((height, width))
    
    for i, cell in enumerate(cells):
        # Get row and col of cell, used to determine position in main_img
        row, col = np.array(divmod(i, 9))
        # Make a copy of the cell image and place it on main_img
        cell_image = cells[i]['img'].copy()
        main_img[row*28:(row+1)*28, col*28:(col+1)*28] = cell_image
    
    fig, ax = plt.subplots()
    ax.imshow(main_img, cmap='gray')
    plt.show(block=False)


# This was previously called get_cell_predictions
def get_predicted_sudoku_grid(model, cells):
    """
    Classify all 81 cells with a ten-class CNN.

    Class mapping:
        0 = blank cell
        1..9 = Sudoku digits
    """
    if len(cells) != 81:
        raise ValueError(
            f"Expected 81 Sudoku cells, received {len(cells)}."
        )

    cell_images = np.array(
        [np.expand_dims(cell["img"], axis=-1) for cell in cells],
        dtype="float32",
    )

    while cell_images.ndim > 4 and cell_images.shape[-1] == 1:
        cell_images = np.squeeze(cell_images, axis=-1)

    if cell_images.shape != (81, 28, 28, 1):
        raise ValueError(
            "Expected input shape (81, 28, 28, 1), "
            f"received {cell_images.shape}."
        )

    if cell_images.size and float(cell_images.max()) > 1.0:
        cell_images /= 255.0

    probabilities = model.predict(cell_images, verbose=0)

    if probabilities.ndim != 2 or probabilities.shape[1] != 10:
        raise ValueError(
            "A ten-class model is required. Expected output shape "
            f"(N, 10), received {probabilities.shape}."
        )

    predicted_labels = np.argmax(probabilities, axis=1).astype(int)

    for index, (cell, predicted_label) in enumerate(
        zip(cells, predicted_labels)
    ):
        cell["contains_digit"] = bool(predicted_label != 0)
        cell["predicted_class"] = int(predicted_label)
        cell["prediction_confidence"] = float(
            probabilities[index, predicted_label]
        )

    return predicted_labels.reshape((9, 9))


def generate_solution_image(full_image, board_image, cells_list, solved_board_arr, M_matrix):
    '''
    Annotate the original image of the sudoku board with the solution.
    
    Args:
        full_image:
        board_image:
        cells_list:
        solved_board_arr:
        M_matrix:
        
    Returns:
        annotated: Original RGB puzzle image, annotated with solutions.
    
    '''
    
    # Specify the font used for annotating the solutions
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Create a white grid the same shape as the warped image, to place solutions
    solution_img = np.ones_like(board_image) * 255
    # Flatten the solved 2D board array
    flattened_board_array = solved_board_arr.reshape((-1))
    
    # Place the solved digits from blank puzzle cells on the blank white image
    for i, cell in enumerate(cells_list):
        if not cell['contains_digit']:
            # Get cell centroids for text positioning
            x_pos = cell['x_centroid']
            y_pos = cell['y_centroid']
            text = str(flattened_board_array[i])
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            # Specify placement of text
            text_x = int((x_pos - textsize[0] / 2))
            text_y = int((y_pos + textsize[1] / 2))
            # Annotate number with black text
            cv2.putText(solution_img, text, (text_x, text_y), font, 1.3, (0, 0, 0), 2)
    
    # Apply the inverse perspective transform to the solution image
    unwarped_img = cv2.warpPerspective(
        solution_img,
        M_matrix,
        (full_image.shape[1], full_image.shape[0]),
        flags=cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    # Copy original image to annotate with the solution digits
    annotated = full_image.copy()
    # Locate black pixels (number text) and change colour to red
    annotated[np.where(unwarped_img[:,:,0] == 0)] = (255, 15, 0)
    
    return annotated
