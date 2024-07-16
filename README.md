# Exercise Tracking with OpenCV and MediaPipe

This project utilizes OpenCV and MediaPipe to track exercise movements, specifically focusing on counting repetitions of exercises such as curls. The application uses computer vision techniques to detect human poses and calculate joint angles, providing real-time feedback on exercise performance.

## Usage

To start the exercise tracking application, run the following command:
```sh
python main.py
```

Ensure that your webcam is connected, as the application uses the webcam feed to detect and track exercises.

## Functions

### calculate_angle

```python
def calculate_angle(a, b, c):
    """
    Calculates the angle between three points.

    Args:
        a, b, c: Coordinates of the points.

    Returns:
        angle: The angle in degrees.
    """
```

### calculate_position

```python
def calculate_position(x, y):
    """
    Calculates the position for displaying the angle text.

    Args:
        x (int): X-coordinate.
        y (int): Y-coordinate.

    Returns:
        tuple: Position coordinates.
    """
```

### get_joint_position

```python
def get_joint_position(landmarks, joint_name):
    """
    Retrieves the position of a specific joint.

    Args:
        landmarks: Detected landmarks.
        joint_name (str): Name of the joint.

    Returns:
        tuple: Joint coordinates.
    """
```

### exercise_counter

```python
def exercise_counter(exercise, angle, count, stage):
    """
    Counts the number of exercise repetitions.

    Args:
        exercise (str): Name of the exercise.
        angle (float): Angle of the joint.
        count (int): Current count of repetitions.
        stage (str): Current stage of the exercise.

    Returns:
        tuple: Updated count and stage.
    """
```

### draw_angle

```python
def draw_angle(image, angle, position):
    """
    Draws the angle on the image.

    Args:
        image: The image frame.
        angle (float): The angle to draw.
        position (tuple): Position to draw the angle.
    """
```

### draw_reps_counter

```python
def draw_reps_counter(image, count):
    """
    Draws the repetition counter on the image.

    Args:
        image: The image frame.
        count (int): Number of repetitions.
    """
```
