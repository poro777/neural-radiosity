import mitsuba as mi
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import drjit as dr
import cv2

mi.set_variant('cuda_ad_rgb')

from nerad.utils.render_utils import process_nerad_output

class render_system():
    '''
        q           : quit
        drag mouse  : change direction
        w,a,s,d     : move
        r           : reset pos and dir
        t,y         : y-axis
        i,o,k,l     : change direction
        z           : save image

    '''
    def __init__(self, scene, origin, angle, integrator = None) -> None:
        self.window = "Image"

        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, render_system.mouse_callback, self)

        self.mode = "path"
        self.nerad = integrator is not None

        cv2.createTrackbar("SSP", self.window, 1, 256, lambda x: render_system.on_SSP_change(x, self))
        cv2.createTrackbar("Resolution", self.window, 5, 10, lambda x: render_system.on_Resolution_change(x, self))
        cv2.createTrackbar("Depth", self.window, 1, 16, lambda x: render_system.on_Depth_change(x, self))
        cv2.createTrackbar("Mode", self.window, 1, 3, lambda x: render_system.on_Mode_change(x, self))

        resoultion = 6
        self.ssp = 32
        self.depth = 8


        self.width = 2 ** resoultion

        self.height = self.width
        self.sensor = None
        self.intergrator = integrator
        self.params = None


        self.change = False
        self.pre_x, self.pre_y = 0,0
        self.scene = scene
        self.origin = np.array(origin).astype(float)
        self.angle = np.array(angle).astype(float)
        self.positive_y = np.array([0,1,0]).astype(float)

        cv2.setTrackbarPos("SSP", self.window, self.ssp)
        cv2.setTrackbarPos("Resolution", self.window, resoultion)
        cv2.setTrackbarPos("Depth", self.window, self.depth)

    def mouse_callback(event, x, y, flags, self):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pre_x, self.pre_y = x, y

        if flags == cv2.EVENT_FLAG_LBUTTON:
            # Print the coordinates when left mouse button is clicked
            offset_x, offset_y = x - self.pre_x, y - self.pre_y
            self.angle += np.array([-offset_x, -offset_y]) * 0.003
            self.pre_x, self.pre_y = x, y
            self.change = True

    def on_SSP_change(val, self):
        self.ssp = val
        self.change = True

    def on_Mode_change(val, self):
        if self.nerad:
            return

        if val == 1:
            self.mode = "path"
        elif val == 2:
            self.mode = "ptracer"


        self.intergrator = mi.load_dict({
            'type' : self.mode,
            'max_depth': self.depth
        })
        self.change = True

    def on_Depth_change(val, self):
        if self.nerad:
            return

        self.depth = val
        self.intergrator = mi.load_dict({
            'type' : self.mode,
            'max_depth': self.depth
        })
        self.change = True

    def on_Resolution_change(val, self):
        self.width = 2**val
        self.height = self.width

        self.change = True
        self.sensor = mi.load_dict({
            'type': 'perspective',
            'fov': 35,
            'to_world': mi.ScalarTransform4f(),
            'sampler_id':{
                'type': 'independent'
            },
            'film_id': {
                'type': 'hdrfilm',
                'width': self.width,
                'height': self.height,
                'pixel_format': 'rgba',
                "filter": {"type": "box"},
            }
        })
        self.params = mi.traverse(self.sensor)

    def getDirection(self):
        cos1 = np.cos(self.angle[1])
        return np.array([cos1 * np.sin(self.angle[0] - np.pi),
                        np.sin(self.angle[1]),
                        cos1 * np.cos(self.angle[0] - np.pi)])

    def getTarget(self):
        return self.origin + self.getDirection()

    def main(self):
        windowWidth, windowHeight = 512, 512

        target = self.getTarget()

        try:
            cv2.imshow(self.window, np.zeros((windowWidth, windowHeight)))

            while True:
                key = cv2.waitKey(1000//30) & 0xFF

                if key == ord('w'):
                    self.origin += self.getDirection() * 0.2
                elif key == ord('s'):
                    self.origin -= self.getDirection() * 0.2

                elif key == ord('a'):
                    self.origin -= np.cross(self.getDirection(), self.positive_y) * 0.2
                elif key == ord('d'):
                    self.origin += np.cross(self.getDirection(), self.positive_y) * 0.2

                elif key == ord('r'):
                    self.origin = np.zeros_like(self.origin)
                    self.angle = np.zeros_like(self.angle)

                elif key == ord('t'):
                    self.origin += self.positive_y * 0.2
                elif key == ord('y'):
                    self.origin -= self.positive_y * 0.2

                elif key == ord('i'):
                    self.angle[0] += 0.05
                elif key == ord('o'):
                    self.angle[0] -= 0.05

                elif key == ord('k'):
                    self.angle[1] += 0.05
                elif key == ord('l'):
                    self.angle[1] -= 0.05

                elif key == ord('q') or not cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) >= 1:
                    break
                elif key == ord('z'):
                    #TODO
                    continue
                elif not self.change:
                    continue

                self.change = False
                view = mi.ScalarTransform4f.look_at(self.origin, self.getTarget(), self.positive_y)
                self.params['to_world'] = view
                self.params.update()

                image = mi.render(self.scene, spp=self.ssp, integrator = self.intergrator, sensor=self.sensor)
                if self.nerad:
                    _, LHS, RHS = process_nerad_output(image)
                    image = LHS
                cv2.imshow(self.window, cv2.resize(cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR),
                                               (windowWidth, windowHeight),
                                               interpolation= cv2.INTER_NEAREST ))
            print( f'look_at: \npos = {self.origin}\ndir = {self.getTarget()} (self.angle = {self.angle})')

        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    scene = "./data/NeRad_paper_scenes/veach_ajar/scene.xml"
    scene = mi.load_file(scene)
    sys = render_system(scene, [0,0,0],[0,0])
    sys.main()
