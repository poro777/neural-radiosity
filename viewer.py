import mitsuba as mi
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import drjit as dr
import cv2
import torch
import argparse

mi.set_variant('cuda_ad_rgb')

def process_nerad_output(img):
    residual = img[:, :, -3:]
    LHS = img[:, :, -7:-3]
    RHS = img[:, :, :-7]
    return residual, LHS, RHS

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

        self.nerad = integrator is not None
        self.mode = "LHS" if self.nerad else "path"

        self.limit = (2 ** 20) if self.nerad else (2 ** 28) # limit ssp per frame (ssp * width * height)

 
        resoultion = 8
        self.ssp = 8
        self.depth = 8

        self.stop = 0
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

        cv2.createTrackbar("Disable", self.window, 0, 1, lambda x: render_system.on_state_change(x, self))
        cv2.createTrackbar("SSP", self.window, self.ssp, 256, lambda x: render_system.on_SSP_change(x, self))
        cv2.createTrackbar("Resolution", self.window, resoultion, 10, lambda x: render_system.on_Resolution_change(x, self))
        cv2.createTrackbar("Depth", self.window, self.depth, 16, lambda x: render_system.on_Depth_change(x, self))
        cv2.createTrackbar("Mode", self.window, 0, 1, lambda x: render_system.on_Mode_change(x, self))

        cv2.setTrackbarMin('SSP', self.window, 1) 
        cv2.setTrackbarMin('Resolution', self.window, 1) 
        cv2.setTrackbarMin('Depth', self.window, 1) 

    def on_state_change(val, self):
        self.stop = val

    def mouse_callback(event, x, y, flags, self):
        if self.stop:
            return
        
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
            if val == 0:
                self.mode = "LHS"
            elif val == 1:
                self.mode = "RHS"
            self.change = True
            return
        
        if val == 0:
            self.mode = "path"
        elif val == 1:
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
                
                if key == ord('q') or not cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) >= 1:
                    break

                elif self.stop:
                    continue
                elif (self.ssp * self.height * self.width) > self.limit:
                    print(f'{self.ssp} * {self.height} * {self.width} > limit:{self.limit}')
                    cv2.setTrackbarPos("Disable", self.window, 1)
                    continue

                elif key == ord('w'):
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

                elif key == ord('z'):
                    cv2.imwrite("./output.png", image)
                    print("save ./output.png")
                    continue
                elif not self.change:
                    continue

                self.change = False
                view = mi.ScalarTransform4f.look_at(self.origin, self.getTarget(), self.positive_y)
                self.params['to_world'] = view
                self.params.update()

                with dr.suspend_grad():
                    with torch.no_grad():
                        image = mi.render(self.scene, spp=self.ssp, integrator = self.intergrator, sensor=self.sensor)
                        if self.nerad:
                            _, LHS, RHS = process_nerad_output(image)

                            image = RHS if self.mode == "RHS" else LHS 
                        image = cv2.cvtColor(np.array( mi.Bitmap(image).convert(mi.Bitmap.PixelFormat.RGBA,mi.Struct.Type.UInt8,True)), cv2.COLOR_RGBA2BGR)

                cv2.imshow(self.window, cv2.resize(image,
                                               (windowWidth, windowHeight),
                                               interpolation= cv2.INTER_NEAREST ))
            print( f'look_at: \npos = {self.origin}\ndir = {self.getTarget()} (self.angle = {self.angle})')

        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--path",default='./data/NeRad_paper_scenes/veach_ajar/scene.xml', help="path", type=str)

    # Parse the arguments
    args = parser.parse_args()

    scene = args.path
    scene = mi.load_file(scene)
    sys = render_system(scene, [0,0,0],[0,0])
    sys.main()
