# Part 2 - Sudoku digit recognition and performance assessment

Alternative to the second part of the [original statement](../statement.pdf).

## Tasks

1. Achieve tasks *2.1 Performance assessment of line segment detection* and *2.3 Image annotation* (only annotate lines) of the original statement.

2. Build a database of Sudoku images with printed and handwritten digits. The size of the database should be sufficient to train a network and assess its performances.

3. Using Cytomine, annotate the Sudoku grids. Also implement a way to annotate digits.

4. Implement a system of the following architecture :
	* Intup : a Sudoku grid picture;
	* Output : a `9` by `9` array filled with recognized digits;
	* The system *must* use machine learning approaches to recognize the digits.

> It is expected to consider, test and evaluate several (at least `2`) different approaches.

5. Define a performance criteria and assess the system on the database.

## Presentation

* Present the chosen performance criteria and the results for the assessment of the line detection of the first part;
* Present the architecture of the implemented system;
* Present the chosen performance criteria and the results obtained on the database;
* Demonstrate a *product* prototype that would be able to recognize printed and handwritten digits on a *live picture* of a Sudoku grid.
