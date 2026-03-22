import numpy as np
import matplotlib.pyplot as plt


class TrackGenerator:
    def __init__(self, separation_dist=0.01, target_speed=10):
        """
        Initialize the track generator with default separation distance and target speed.
        """
        self.separation_dist = separation_dist  # Distance between points on the path
        self.target_speed = target_speed  # Target speed along the track
        self.track_segments = []  # List to hold index ranges of each segment
        self.xr = []  # X coordinates of the track centerline
        self.yr = []  # Y coordinates of the track centerline

    def smooth_transition(self, start_angle, end_angle, count):
        """
        Generate a smoothly interpolated array of angles from start to end.
        """
        return np.linspace(start_angle, end_angle, count)

    def calculate_count(self, length):
        """
        Calculate number of points needed based on path length and separation distance.
        """
        return int(np.ceil(length / self.separation_dist))

    def straight(self, init_coord, end_coord, init_angle):
        """
        Generate a straight-line segment between two coordinates.
        """
        length = np.linalg.norm(np.array(end_coord) - np.array(init_coord))
        count = self.calculate_count(length)

        xr = np.linspace(init_coord[0], end_coord[0], count)
        yr = np.linspace(init_coord[1], end_coord[1], count)
        thetar = np.full(count, init_angle)  # Constant orientation
        kr = np.zeros(count)  # Zero curvature for straight lines

        return xr, yr, thetar, kr

    def arc(self, init_coord, end_coord, init_angle, end_angle):
        """
        Generate an arc segment from initial to final coordinate, with specified angles.
        """
        L = np.linalg.norm(np.array(end_coord) - np.array(init_coord))
        R = L / np.sqrt(2 * (1 - np.cos(end_angle - init_angle)))  # Radius of curvature
        arc_length = abs(R * (end_angle - init_angle))
        count = self.calculate_count(arc_length)

        delta_angle = (end_angle - init_angle) / (count - 1)
        thetar = self.smooth_transition(init_angle, end_angle, count)

        xr = np.zeros(count)
        yr = np.zeros(count)
        kr = np.full(count, 1 / R if delta_angle > 0 else -1 / R)  # Constant curvature

        for i in range(count):
            angle = init_angle + delta_angle * i
            xr[i] = init_coord[0] - R * np.sin(init_angle) + R * np.sin(angle)
            yr[i] = init_coord[1] + R * np.cos(init_angle) - R * np.cos(angle)

        return xr, yr, thetar, kr

    def generate_track(self):
        """
        Generate a full track consisting of multiple straight and arc segments.
        """
        # Segment 1: Straight
        x1, y1, theta1, kr1 = self.straight([0, 0], [0, 7.5], np.pi / 2)

        # Segment 2: Left arc
        x2, y2, theta2, kr2 = self.arc([0, 7.5], [-5, 12.5], np.pi / 2, np.pi)

        # Segment 3: Straight
        x3, y3, theta3, kr3 = self.straight([-5, 12.5], [-12.5, 12.5], np.pi)

        # Segment 4: Right arc
        x4, y4, theta4, kr4 = self.arc([-12.5, 12.5], [-17.5, 7.5], np.pi, 3 * np.pi / 2)

        # Segment 5: Final straight
        x5, y5, theta5, kr5 = self.straight([-17.5, 7.5], [-17.5, 0], 3 * np.pi / 2)

        # Concatenate all segments
        self.xr = np.concatenate([x1, x2, x3, x4, x5]).tolist()
        self.yr = np.concatenate([y1, y2, y3, y4, y5]).tolist()
        thetar = np.concatenate([theta1, theta2, theta3, theta4, theta5]).tolist()
        kappar = np.concatenate([kr1, kr2, kr3, kr4, kr5]).tolist()
        sp = [self.target_speed] * len(self.xr)  # Constant speed for each point

        # Save segment index ranges
        self.track_segments = [
            (0, len(x1)),
            (len(x1), len(x1) + len(x2)),
            (len(x1) + len(x2), len(x1) + len(x2) + len(x3)),
            (len(x1) + len(x2) + len(x3), len(x1) + len(x2) + len(x3) + len(x4)),
            (len(x1) + len(x2) + len(x3) + len(x4), len(self.xr))
        ]

        # Compute left and right boundaries (0.5 m offset)
        self.x_left = self.xr + np.sin(thetar) * 0.5
        self.y_left = self.yr - np.cos(thetar) * 0.5
        self.x_right = self.xr - np.sin(thetar) * 0.5
        self.y_right = self.yr + np.cos(thetar) * 0.5

        return self.xr, self.yr, thetar, kappar, sp, self.x_left, self.x_right, self.y_left, self.y_right

    def get_current_segment(self, car_x, car_y):
        """
        Determine the current segment index based on the car's position.
        """
        min_dist = float('inf')
        segment_index = -1

        for i, (start_idx, end_idx) in enumerate(self.track_segments):
            for j in range(start_idx, end_idx):
                dist = np.hypot(self.xr[j] - car_x, self.yr[j] - car_y)
                if dist < min_dist:
                    min_dist = dist
                    segment_index = i

        return segment_index

    def visualize_track(self):
        """
        Visualize the track and its boundaries using matplotlib.
        """
        plt.figure()
        axes = plt.gca()

        plt.plot(self.xr, self.yr, label="Center Line")
        plt.plot(self.x_left, self.y_left, label="Left Boundary", color='red', linestyle='--')
        plt.plot(self.x_right, self.y_right, label="Right Boundary", color='green', linestyle='--')

        axes.set_xlim([-40, 10])
        axes.set_ylim([-20, 50])
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Generated Track with Boundaries")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # Create a track generator instance
    track_gen = TrackGenerator(separation_dist=0.1, target_speed=3.6)

    # Generate the track
    track_gen.generate_track()

    # Visualize the track
    track_gen.visualize_track()
