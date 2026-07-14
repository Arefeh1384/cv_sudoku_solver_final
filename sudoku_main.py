import argparse
import os
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import sudoku_utils as sutils
from sudoku_solver_class import SudokuSolver


def solve_sudoku_puzzle(args):
    img_fpath = args["img_fpath"]
    model_fpath = args["model_fpath"]

    if not os.path.exists(img_fpath):
        raise FileNotFoundError(f"File not found: '{img_fpath}'")
    if not os.path.exists(model_fpath):
        raise FileNotFoundError(f"Model not found: '{model_fpath}'")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = Path(args["debug_root"]) / run_id
    debug_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(img_fpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = sutils.resize_and_maintain_aspect_ratio(img, new_width=1000)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    plt.tight_layout()
    plt.show(block=False)

    loaded_model = tf.keras.models.load_model(model_fpath, compile=False)

    diagnostics = {
        "status": "runtime_failed",
        "reason": "Unexpected runtime failure.",
        "debug_directory": str(debug_dir),
    }

    try:
        cells, M, board_image, diagnostics = sutils.get_valid_cells_from_image(
            img,
            debug_dir=debug_dir,
            return_diagnostics=True,
        )

        grid_array = sutils.get_predicted_sudoku_grid(loaded_model, cells)
        np.savetxt(debug_dir / "09_predicted_grid.txt", grid_array, fmt="%d")

        solver = SudokuSolver(board=grid_array.copy())
        solved = solver.solve()

        if solved and not np.any(np.asarray(solver.board) == 0):
            diagnostics["status"] = "success"
            diagnostics["reason"] = "Grid extraction, digit recognition and solving succeeded."
            print("Success - sudoku solved!")

            annotated = sutils.generate_solution_image(
                img, board_image, cells, np.asarray(solver.board), M
            )
            cv2.imwrite(
                str(debug_dir / "10_solution.png"),
                cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
            )

            fig, ax = plt.subplots()
            ax.imshow(annotated)
            ax.axis("off")
            plt.tight_layout()
            plt.show(block=False)
        else:
            diagnostics["status"] = "solver_failed"
            diagnostics["reason"] = (
                "The 81 cells were extracted, but the predicted digits produced "
                "an invalid or unsolvable Sudoku. One or more digits were probably misclassified."
            )
            print("Could not solve the puzzle. Check for misclassified digits.\n")

        solver.print_board()

    except sutils.SudokuExtractionError as exc:
        diagnostics = exc.diagnostics
        print("Sudoku extraction failed:")
        print(diagnostics.get("reason", str(exc)))

    except Exception as exc:
        diagnostics["status"] = "runtime_failed"
        diagnostics["reason"] = f"{type(exc).__name__}: {exc}"
        raise

    finally:
        sutils.save_diagnostic_json(debug_dir / "report.json", diagnostics)
        sutils.append_diagnostic_report(
            Path(args["debug_root"]) / "report.csv",
            img_fpath,
            diagnostics,
        )
        print(f"Diagnostic files saved in: {debug_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--img_fpath",
        default="data/sudoku_images/22.jpg",
        type=str,
        help="Path to sudoku image file",
    )
    ap.add_argument(
    "--model_fpath",
    default="models/model_phase2_10class_transfer.keras",
    type=str,
    help="Path to the ten-class Keras CNN model",
    )
    ap.add_argument(
        "--debug_root",
        default="debug_runs",
        type=str,
        help="Directory used for diagnostic images and reports",
    )
    args = vars(ap.parse_args())

    solve_sudoku_puzzle(args)
    plt.show()
