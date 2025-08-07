import cv2
import numpy as np

import cv2
import os

print("cv2 version:", cv2.__version__)
print("cv2 file path:", cv2.__file__)
print("Has CharucoBoard_create:", hasattr(cv2.aruco, "CharucoBoard_create"))


def create_charuco_board(
    squares_x=5,
    squares_y=7,
    square_length=0.04,
    marker_length=0.02,
    dictionary=cv2.aruco.DICT_5X5_100
):
    # Load ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)

    # Create Charuco board object
    board = cv2.aruco.CharucoBoard_create(
        squares_x, squares_y, square_length, marker_length, aruco_dict
    )

    return board

def draw_board_image(board, img_size_pixels=(600, 500), margin=10, border_bits=1):
    # Draw the board into an image
    img = board.draw(img_size_pixels, marginSize=margin, borderBits=border_bits)
    return img

if __name__ == "__main__":
    # Parameters (adjust as needed)
    squares_x = 5         # number of chessboard squares in X
    squares_y = 7         # number in Y
    square_length = 0.04  # e.g. 4 cm
    marker_length = 0.02  # e.g. 2 cm
    img_size = (800, 600) # output image size in pixels
    margin = 20           # pixel margin
    border_bits = 1       # marker border width

    board = create_charuco_board(
        squares_x,
        squares_y,
        square_length,
        marker_length,
        cv2.aruco.DICT_5X5_100
    )

    img = draw_board_image(board, img_size, margin, border_bits)

    # Save and optionally display
    cv2.imwrite("charuco_board.png", img)
    print("Saved charuco_board.png")
    # Uncomment to show window
    # cv2.imshow("ChArUco Board", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
