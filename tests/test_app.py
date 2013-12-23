import unittest

from app import Config

class TestConfig(unittest.TestCase):
    def test(self):
        """
        The Config class should able to take a dictionary with values, lists,
        and nested dictionaries, and attach them as attributes to the Config
        object.

        """
        config = Config({
            'app': {
                'window': {
                    'height': 600,
                    'width': 800
                    },
                'shaders': ['basic.frag', 'basic.vert']
                }
            })

        self.assertEqual(600, config.app.window.height)
        self.assertEqual(800, config.app.window.width)
        self.assertIn('basic.frag', config.app.shaders)
        self.assertIn('basic.vert', config.app.shaders)


if __name__ == "__main__":
    unittest.main()
