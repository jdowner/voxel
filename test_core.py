import math
import unittest

import numpy

from core import (Camera, Quaternion)

class TestQuaternion(unittest.TestCase):
    def setUp(self):
        self.a = Quaternion(1,0,1,0)
        self.b = Quaternion(1,1,0,0)
        self.c = Quaternion(0,1,0,1)

    def test_iadd(self):
        self.a += self.c
        self.b += self.c
        self.assertEqual(Quaternion(1,1,1,1), self.a)
        self.assertEqual(Quaternion(1,2,0,1), self.b)

    def test_isub(self):
        self.a -= self.c
        self.b -= self.c
        self.assertEqual(Quaternion(1,-1,1,-1), self.a)
        self.assertEqual(Quaternion(1,0,0,-1), self.b)

    def test_imul(self):
        ua = numpy.float32(self.a.w)
        ub = numpy.float32(self.b.w)
        va = numpy.array([self.a.x, self.a.y, self.a.z])
        vb = numpy.array([self.b.x, self.b.y, self.b.z])

        u = ua * ub - numpy.dot(va, vb)
        v = ua * vb + ub * va + numpy.cross(va, vb)
        self.a *= self.b

        self.assertAlmostEqual(self.a.w, u)
        self.assertAlmostEqual(self.a.x, v[0])
        self.assertAlmostEqual(self.a.y, v[1])
        self.assertAlmostEqual(self.a.z, v[2])

    def test_add(self):
        self.assertEqual(Quaternion(1,1,1,1), self.a + self.c)
        self.assertEqual(Quaternion(1,2,0,1), self.b + self.c)

    def test_sub(self):
        self.assertEqual(Quaternion(1,-1,1,-1), self.a - self.c)
        self.assertEqual(Quaternion(1,0,0,-1), self.b - self.c)

    def test_mul(self):
        ua = numpy.float32(self.a.w)
        ub = numpy.float32(self.b.w)
        va = numpy.array([self.a.x, self.a.y, self.a.z])
        vb = numpy.array([self.b.x, self.b.y, self.b.z])

        u = ua * ub - numpy.dot(va, vb)
        v = ua * vb + ub * va + numpy.cross(va, vb)
        r = self.a * self.b

        self.assertAlmostEqual(r.w, u)
        self.assertAlmostEqual(r.x, v[0])
        self.assertAlmostEqual(r.y, v[1])
        self.assertAlmostEqual(r.z, v[2])

    def test_eq(self):
        self.assertTrue(Quaternion(1,2,3,4), Quaternion(1,2,3,4))

    def test_ne(self):
        self.assertTrue(Quaternion(1,2,3,4), Quaternion(0,2,3,4))

    def test_length(self):
        self.assertAlmostEqual(math.sqrt(2.0), self.a.length, places=6)
        self.assertAlmostEqual(math.sqrt(2.0), self.b.length, places=6)
        self.assertAlmostEqual(math.sqrt(2.0), self.c.length, places=6)

    def test_normalize(self):
        self.assertTrue(self.a.length > 1.0)
        self.assertTrue(self.b.length > 1.0)
        self.assertTrue(self.c.length > 1.0)
        self.assertAlmostEqual(1.0, self.a.normalized().length, places=6)
        self.assertAlmostEqual(1.0, self.b.normalized().length, places=6)
        self.assertAlmostEqual(1.0, self.c.normalized().length, places=6)

    def test_inverted(self):
        self.assertEqual(self.a, self.a.inverted().inverted())
        self.assertAlmostEqual( 0.5, self.a.inverted().w)
        self.assertAlmostEqual( 0.0, self.a.inverted().x)
        self.assertAlmostEqual(-0.5, self.a.inverted().y)
        self.assertAlmostEqual( 0.0, self.a.inverted().z)
        self.assertTrue(self.a is not self.a.inverted().inverted())

    def test_from_axis_angle(self):
        theta = math.pi / 5.0

        c = math.cos(theta / 2.0)
        s = math.sin(theta / 2.0)
        q = Quaternion.from_axis_angle(numpy.array([1, -1, 1]), theta)

        self.assertAlmostEqual( c, q.w)
        self.assertAlmostEqual( s * math.sqrt(1.0 / 3.0), q.x)
        self.assertAlmostEqual(-s * math.sqrt(1.0 / 3.0), q.y)
        self.assertAlmostEqual( s * math.sqrt(1.0 / 3.0), q.z)

    def test_rotate(self):
        q = Quaternion.from_axis_angle(numpy.array([0, 1, 0]), math.pi / 2.0)
        v = q.rotate((1, 0, 0))

        self.assertAlmostEqual( 0, v[0])
        self.assertAlmostEqual( 0, v[1])
        self.assertAlmostEqual(-1, v[2])

    def test_conjugated(self):
        self.assertEqual(Quaternion(1, 0, -1, 0), self.a.conjugated())
        self.assertEqual(Quaternion(1, -1, 0, 0), self.b.conjugated())
        self.assertEqual(Quaternion(0, -1, 0, -1), self.c.conjugated())


class TestCamera(unittest.TestCase):
    def setUp(self):
        self.camera = Camera()

        self.e1 = numpy.array([1,0,0])
        self.e2 = numpy.array([0,1,0])
        self.e3 = numpy.array([0,0,1])

    def test_initialization(self):
        self.assertTrue(numpy.allclose(self.e1, self.camera.right))
        self.assertTrue(numpy.allclose(self.e2, self.camera.up))
        self.assertTrue(numpy.allclose(self.e3, self.camera.backward))

    def test_rotation(self):
        self.camera.yaw(math.pi / 120.0)

        angle = self.camera.orientation.angle()
        axis = self.camera.orientation.axis()

        self.assertAlmostEqual(math.pi / 120.0, angle, places = 3)
        self.assertTrue(numpy.allclose(self.e2, axis))

    def test_move_forward(self):
        self.camera.position = self.e3
        self.camera.yaw(-math.pi / 4.0)
        self.camera.move_forward(math.sqrt(2.0))

        self.assertTrue(numpy.allclose(self.e1, self.camera.position, atol=1e-6))

    def test_rotation_accumulation(self):
        self.camera.move_backward(1.0)
        self.camera.move_right(1.0)
        self.camera.move_up(1.0)

        for _ in xrange(27):
            self.camera.yaw(math.pi / 12.0)

        print('right',self.camera.right)
        print('up',self.camera.up)
        print('forward',self.camera.forward)
        #self.assertTrue(numpy.allclose(self.e1, self.camera.right))

        self.camera.move_right(math.sqrt(2.0))

        print(self.camera.position)

        print(self.camera.orientation.angle())
        print(self.camera.orientation.axis())



if __name__ == "__main__":
    unittest.main()
