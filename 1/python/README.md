# Part 1 - Main modules development and line extraction

## Authors

* **Maxime Meurisse**
* **François Rozet**
* **Océane Rumfels**
* **Valentin Vermeylen**

## Language

* **Python**

## Implemented files

* `main.py`
* `soccer.py`
* `sudoku.py`
* `utils.py`

## Required libraries

* `os` (native)
* `glob` (native)
* `numpy` (to install - `pip install numpy`)
* `OpenCV` (to install - `pip install opencv-python`)
* `mimetypes` (to install - `pip install mimetypes`)

## Organization of the code

In order to handle sudoku, road and soccer images, we have respectively implemented the files `sudoku.py`, `road.py` and `soccer.py`, each containing a class `Process`.
Each `Process` class is constructed as follow :

* a `preprocessing` method that filters a given image;
* an `edge_points_extraction` method that extracts edge points from a preprocessed image;
* a `line_detection` method that detects lines from extracted edges.

All processing parameters are built-in in the classes.

Theses classes are imported and used in the main method of the file `main.py`.
Also, few auxiliary functions (mostly manipulating path and directories) used in `main.py` are implemented in the file `utils.py`.

## Execute the code

Five elements have to be set in `main.py` for it to run properly :

* `from sudoku import Process` : import the `Process` class from the `sudoku.py` file;
> To process other type of images, you may change `sudoku` by other available packages (`sudoku`, `road` or `soccer`).
* `ORIGIN` variable : the path to a file (image or video) or a directory of images to process;
* `DESTINATION` variable : the destination path to export all processed elements;
> If the path doesn't exist, it will be automatically created.
* `MAKE_VIDEO` variable : a boolean to decide whether or not the program will create a video with highlighted images;
* `FRAME_RATE` variable : if the program creates a video, the frame rate of the video.

Then, to execute the script :

```bash
python main.py
```
