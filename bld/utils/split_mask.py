import numpy as np
import cv2 as cv


class MaskSplitter:
    def __init__(self, im_slice):
        self.min_area = 50
        self.max_dist_ratio = 0.5
        self.slice = im_slice
        self.thresh = self.slice.copy().astype(np.uint8)
        self.splitted = self.thresh.copy()

    def run(self):
        # cv.findContours(
        #     image, mode, method[, contours[, hierarchy[, offset]]]
        # ) ->	image, contours, hierarchy
        contours, hierarchy = cv.findContours(self.thresh, 2, 1)

        for contour in contours:
            self.run_for_one_contour(contour=contour)

    def run_for_one_contour(self, contour):
        # cv.convexHull(
        #     points[, hull[, clockwise[, returnPoints]]]
        # ) ->	hull
        hull = cv.convexHull(contour, returnPoints=False)
        hullpoints = contour[hull].squeeze()
        if len(hullpoints) > 2:
            area = cv.contourArea(hullpoints)
        else:
            area = 0

        if cv.contourArea(contour) > self.min_area and area / cv.contourArea(contour) > 1.2:
            # cv.convexityDefects (contour, convexhull, convexityDefect)
            #     convexityDefect:	the output vector of convexity defects.
            #     (start_index_of_hull, end_index_of_hull, farthest_pt_index, max_depth)
            defects = cv.convexityDefects(contour, hull)
            if defects is not None:
                self.apply_split(contour=contour, defects=defects)

    def apply_split(self, contour, defects):
        farthest_points = defects[:, 0, 2].tolist()
        dist = defects[:, 0, 3].tolist()

        # filter the points
        # the point should be at least ... far compared to the farthest point
        points_filtered = self.filter_points(
            distances=dist,
            far_points=farthest_points
        )

        if len(points_filtered) > 1:
            # we find the start and end points for the split line
            end, start = self.find_start_and_end_points(
                points_filtered=points_filtered,
                contour=contour
            )

        # if there is only one very concave area, we have to split with another method
        if len(points_filtered) == 1:
            # this means that the mask is a special case
            print('Special case, it should be evaluated manually.')

    def filter_points(self, distances, far_points):
        points_filtered = []
        max_dist = np.max(distances)
        for j in range(len(distances)):
            if distances[j] / max_dist > self.max_dist_ratio:
                points_filtered.append(far_points[j])
        return points_filtered

    @staticmethod
    def find_start_and_end_points(points_filtered, contour):
        start = None
        end = None
        for k in range(len(points_filtered)):
            f1 = points_filtered[k]
            p1 = tuple(contour[f1][0])
            min_dist = np.inf

            for l in range(len(points_filtered)):
                if k != l:
                    f2 = points_filtered[l]
                    p2 = tuple(contour[f2][0])

                    d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

                    if d < min_dist:
                        min_dist = d
                        end = p2
                        start = p1

        return end, start
