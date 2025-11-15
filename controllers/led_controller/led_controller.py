"""
Simple LED controller for hall world interactive elements.
This controller does nothing but prevents Webots from showing warnings.
"""

from controller import Robot

class LEDController(Robot):
    def __init__(self):
        super().__init__()
        self.timeStep = int(self.getBasicTimeStep())

    def run(self):
        # Simple do-nothing loop
        while self.step(self.timeStep) != -1:
            pass  # Do nothing, just prevent warnings

if __name__ == "__main__":
    controller = LEDController()
    controller.run()