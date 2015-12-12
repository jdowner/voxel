import unittest

from voxel.app import Config

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.config = Config({
            'app': {
                'window': {
                    'height': 600,
                    'width': 800
                    },
                'shaders': ['basic.frag', 'basic.vert']
                }
            })

    def test(self):
        """
        The Config class should able to take a dictionary with values, lists,
        and nested dictionaries, and attach them as attributes to the Config
        object.

        """
        self.assertEqual(600, self.config.app.window.height)
        self.assertEqual(800, self.config.app.window.width)
        self.assertIn('basic.frag', self.config.app.shaders)
        self.assertIn('basic.vert', self.config.app.shaders)

    def test_contains(self):
        """
        This test checks that the __contains__ method works as expected.

        """
        self.assertIn('app', self.config)
        self.assertIn('window', self.config.app)


if __name__ == "__main__":
    unittest.main()
