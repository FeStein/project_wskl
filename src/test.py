import SpatioTemporal.detection as det
import SpatioTemporal.tube as tb

import unittest


class TestIoU(unittest.TestCase):

    def test_corner_case(self):
        """
        Check if ovlerap over corners is correctly calculated
        """

        detA = det.Detection("dummy", 100, 100, 300, 300, 0)

        det1 = det.Detection("dummy", 200, 200, 400, 400, 0)
        det2 = det.Detection("dummy", 0, 200, 200, 400, 0)
        det3 = det.Detection("dummy", 0, 0, 200, 200, 0)
        det4 = det.Detection("dummy", 200, 0, 400, 200, 0)

        gt = 0.14285714285714285
        self.assertEqual(tb.calculate_IOU(detA, det1), gt)
        self.assertEqual(tb.calculate_IOU(detA, det1), gt)
        self.assertEqual(tb.calculate_IOU(detA, det1), gt)
        self.assertEqual(tb.calculate_IOU(detA, det1), gt)

        # check the reverse case
        self.assertEqual(tb.calculate_IOU(det1, detA), gt)
        self.assertEqual(tb.calculate_IOU(det1, detA), gt)
        self.assertEqual(tb.calculate_IOU(det1, detA), gt)
        self.assertEqual(tb.calculate_IOU(det1, detA), gt)

    def test_bound_case(self):
        """
        Check if overlap over boundary is calculated correctly
        """

        detA = det.Detection("dummy", 100, 100, 300, 300, 0)

        det1 = det.Detection("dummy", 200, 150, 400, 250, 0)
        det2 = det.Detection("dummy", 0, 150, 200, 250, 0)
        det3 = det.Detection("dummy", 150, 200, 250, 400, 0)
        det4 = det.Detection("dummy", 150, 0, 250, 200, 0)

        gt = 0.2
        self.assertEqual(tb.calculate_IOU(detA, det1), gt)
        self.assertEqual(tb.calculate_IOU(detA, det1), gt)
        self.assertEqual(tb.calculate_IOU(detA, det1), gt)
        self.assertEqual(tb.calculate_IOU(detA, det1), gt)

        # check the reverse case
        self.assertEqual(tb.calculate_IOU(det1, detA), gt)
        self.assertEqual(tb.calculate_IOU(det1, detA), gt)
        self.assertEqual(tb.calculate_IOU(det1, detA), gt)
        self.assertEqual(tb.calculate_IOU(det1, detA), gt)

    def test_total_overlap(self):
        """
        Check total overlap and no overlap at all
        """

        detA = det.Detection("dummy", 100, 100, 300, 300, 0)

        det1 = det.Detection("dummy", 150, 150, 250, 250, 0)
        det2 = det.Detection("dummy", 0, 0, 400, 400, 0)
        det3 = det.Detection("dummy", 300, 300, 500, 500, 0)

        self.assertEqual(tb.calculate_IOU(detA, det1), 0.25)
        self.assertEqual(tb.calculate_IOU(detA, det2), 0.25)
        self.assertEqual(tb.calculate_IOU(detA, det3), 0.0)

        # check the reverse case
        self.assertEqual(tb.calculate_IOU(det1, detA), 0.25)
        self.assertEqual(tb.calculate_IOU(det2, detA), 0.25)
        self.assertEqual(tb.calculate_IOU(det3, detA), 0.0)


if __name__ == "__main__":
    unittest.main()
