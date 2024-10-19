import os
import cv2
import numpy
from matplotlib.image import imread


class Detection:
    def __init__(self, onnx_path, session=None):
        self.session = session
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            from onnxruntime import InferenceSession
            self.session = InferenceSession(onnx_path,
                                            providers=['CUDAExecutionProvider'])

        self.inputs = self.session.get_inputs()[0]

        self.min_size = 3
        self.max_size = 960
        self.box_thresh = 0.8
        self.mask_thresh = 0.8

        self.mean = numpy.array([123.675, 116.28, 103.53])  # imagenet mean
        self.mean = self.mean.reshape(1, -1).astype('float64')

        self.std = numpy.array([58.395, 57.12, 57.375])  # imagenet std
        self.std = 1 / self.std.reshape(1, -1).astype('float64')

    def filter_polygon(self, points, shape):
        width = shape[1]
        height = shape[0]
        filtered_points = []
        for point in points:
            if type(point) is list:
                point = numpy.array(point)
            point = self.clockwise_order(point)
            w = int(numpy.linalg.norm(point[0] - point[1]))
            h = int(numpy.linalg.norm(point[0] - point[3]))
            if w <= 3 or h <= 3:
                continue
            filtered_points.append(point)

        return numpy.array(filtered_points)

    def boxes_from_bitmap(self, output, mask, dest_width, dest_height):
        mask = (mask * 255).astype(numpy.uint8)
        height, width = mask.shape

        outs = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 2:
            contours = outs[0]
        else:
            contours = outs[1]

        boxes = []
        scores = []
        for index in range(len(contours)):
            contour = contours[index]
            points, min_side = self.get_min_boxes(contour)
            if min_side < self.min_size:
                continue
            points = numpy.array(points)
            score = self.box_score(output, contour)
            if self.box_thresh > score:
                continue

            # box, min_side = self.get_min_boxes(points)
            if min_side < self.min_size + 2:
                continue

            box = numpy.array(points)

            box[:, 0] = numpy.clip(numpy.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = numpy.clip(numpy.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)

        return numpy.array(boxes, dtype="int32"), scores

    @staticmethod
    def get_min_boxes(contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    @staticmethod
    def box_score(bitmap, contour):
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = numpy.reshape(contour, (-1, 2))

        x1 = numpy.clip(numpy.min(contour[:, 0]), 0, w - 1)
        y1 = numpy.clip(numpy.min(contour[:, 1]), 0, h - 1)
        x2 = numpy.clip(numpy.max(contour[:, 0]), 0, w - 1)
        y2 = numpy.clip(numpy.max(contour[:, 1]), 0, h - 1)

        mask = numpy.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=numpy.uint8)

        contour[:, 0] = contour[:, 0] - x1
        contour[:, 1] = contour[:, 1] - y1
        contour = contour.reshape((1, -1, 2)).astype("int32")

        cv2.fillPoly(mask, contour, color=(1, 1))
        return cv2.mean(bitmap[y1:y2 + 1, x1:x2 + 1], mask)[0]

    @staticmethod
    def clockwise_order(point):
        poly = numpy.zeros((4, 2), dtype="float32")
        s = point.sum(axis=1)
        poly[0] = point[numpy.argmin(s)]
        poly[2] = point[numpy.argmax(s)]
        tmp = numpy.delete(point, (numpy.argmin(s), numpy.argmax(s)), axis=0)
        diff = numpy.diff(numpy.array(tmp), axis=1)
        poly[1] = tmp[numpy.argmin(diff)]
        poly[3] = tmp[numpy.argmax(diff)]
        return poly

    @staticmethod
    def clip(points, h, w):
        for i in range(points.shape[0]):
            points[i, 0] = int(min(max(points[i, 0], 0), w - 1))
            points[i, 1] = int(min(max(points[i, 1], 0), h - 1))
        return points

    def resize(self, image):
        h, w = image.shape[:2]

        # limit the max side
        if max(h, w) > self.max_size:
            if h > w:
                ratio = float(self.max_size) / h
            else:
                ratio = float(self.max_size) / w
        else:
            ratio = 1.

        resize_h = max(int(round(int(h * ratio) / 32) * 32), 32)
        resize_w = max(int(round(int(w * ratio) / 32) * 32), 32)

        return cv2.resize(image, (resize_w, resize_h))

    @staticmethod
    def zero_pad(image):
        h, w, c = image.shape
        pad = numpy.zeros((max(32, h), max(32, w), c), numpy.uint8)
        pad[:h, :w, :] = image
        return pad

    def __call__(self, x):
        h, w = x.shape[:2]

        if sum([h, w]) < 64:
            x = self.zero_pad(x)

        x = self.resize(x)
        x = x.astype('float32')

        cv2.subtract(x, self.mean, x)  # inplace
        cv2.multiply(x, self.std, x)  # inplace

        x = x.transpose((2, 0, 1))
        x = numpy.expand_dims(x, axis=0)

        outputs = self.session.run(None, {self.inputs.name: x})[0]
        outputs = outputs[0, 0, :, :]

        boxes, scores = self.boxes_from_bitmap(outputs, outputs > self.mask_thresh, w, h)

        return self.filter_polygon(boxes, (h, w))


image = cv2.imread('test.png')
model = Detection('/Users/arish/Software Development/Email OCR/flutter_app/assets/models/detection.onnx')
result = model(image)


def sort_polygon(points):
    points.sort(key=lambda x: (x[0][1], x[0][0]))
    for i in range(len(points) - 1):
        for j in range(i, -1, -1):
            if abs(points[j + 1][0][1] - points[j][0][1]) < 10 and (points[j + 1][0][0] < points[j][0][0]):
                temp = points[j]
                points[j] = points[j + 1]
                points[j + 1] = temp
            else:
                break
    return points


def crop_image(image, points):
    assert len(points) == 4, "shape of points must be 4*2"
    crop_width = int(max(numpy.linalg.norm(points[0] - points[1]),
                         numpy.linalg.norm(points[2] - points[3])))
    crop_height = int(max(numpy.linalg.norm(points[0] - points[3]),
                          numpy.linalg.norm(points[1] - points[2])))
    pts_std = numpy.float32([[0, 0],
                             [crop_width, 0],
                             [crop_width, crop_height],
                             [0, crop_height]])
    matrix = cv2.getPerspectiveTransform(points, pts_std)
    image = cv2.warpPerspective(image,
                                matrix, (crop_width, crop_height),
                                borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    height, width = image.shape[0:2]
    if height * 1.0 / width >= 1.5:
        image = numpy.rot90(image, k=3)
    return image


for point in result:
    point = numpy.array(point, dtype=numpy.int32)
    cv2.polylines(image,
                  [point], True,
                  (0, 255, 0), 2)

cv2.imwrite('result.png', image)

# print(crop_image(result, sort_polygon(list(result))))


# def compare_files_byte_by_byte(file1_path, file2_path):
#     with open(file1_path, 'rb') as file1, open(file2_path, 'rb') as file2:
#         # Read and compare byte by byte
#         while True:
#             byte1 = file1.read(1)
#             byte2 = file2.read(1)
#             if byte1 != byte2:
#                 return False
#             if not byte1:  # End of file
#                 break
#     return True
#
# # Example usage
# # print(compare_files_byte_by_byte('/Users/arish/Research/research/rna_modification/test2.png', '/Users/arish/Software Development/Email OCR/flutter_app/sample.png'))
# print(compare_files_byte_by_byte('f.fff', 'f1.fff'))
